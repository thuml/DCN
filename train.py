##################################################################################
# 2017 01.16 Created by Shichen Liu                                              #
# Residual Transfer Network implemented by tensorflow                            #
#                                                                                #
#                                                                                #
##################################################################################

import os
import sys
import argparse

LEARNING_RATE = 0.0001
DATA_PATH = '/path/to/dataset/'
BATCH_SIZE = 64
CROSS_ENTROPY_LAMBDA = 1.0
ENTROPY_LAMBDA = 1.0
ENTROPY_LOSS_WEIGHT = 0.0
MANUAL_SEED = 1

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', metavar='DATASET', type=str,
                    choices=['AWA', 'SUN', 'CUB', 'APY'])
parser.add_argument('--data-path', type=str, default=DATA_PATH)
parser.add_argument('--lr', type=float, default=LEARNING_RATE)
parser.add_argument('-bs', '--batch-size', type=int, default=BATCH_SIZE)

parser.add_argument('-cel', '--cross-entropy-lambda', type=float, default=CROSS_ENTROPY_LAMBDA)
parser.add_argument('-el', '--entropy-lambda', type=float, default=ENTROPY_LAMBDA)
parser.add_argument('-elw', '--entropy-loss-weight', type=float, default=ENTROPY_LOSS_WEIGHT)
parser.add_argument('-s', '--manual-seed', type=int, default=MANUAL_SEED)
parser.add_argument('-g', '--gpu', type=str, default='0')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(args)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time
from math import ceil
import random
from utils import *


torch.backends.cudnn.deterministic = True
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
np.random.seed(args.manual_seed)


class Bottleneck(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Bottleneck, self).__init__()
        z_dim = 1024

        self.fc = nn.Linear(x_dim, z_dim)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()

        self.fc1 = nn.Linear(z_dim, y_dim)
        self.fc1.weight.data.normal_(0, 0.03)
        self.fc1.bias.data.zero_()
        
    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = F.dropout(x, inplace=True)

        x = self.fc1(x)
        x = torch.tanh(x)
        return x


def sim(x, y):
    # Inner product similarity
    ip_sim = torch.mm(x, y)
    return ip_sim


class Net(object):
    def __init__(self, config):
        ### Initialize setting
        log('setup', 'Initializing network')
        np.set_printoptions(precision=4)
        self.config = config
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.max_iter = config['max_iter']
        self.n_class = config['n_class']
        self.entropy_lambda = config['entropy_lambda']
        self.cross_entropy_lambda = config['cross_entropy_lambda']
        self.embedding_scale = 10.
        
        self.class_embedding = self.embedding_scale*np.tanh(config['class_embedding'])
        self.part_ids = config['part_ids']

        self.bottleneck_dim = config['class_embedding_dim']
        self.feature_dim = 2048
        self.print_freq = 500

        ### Construct network structure
        log('setup', 'Creating network')
        self.bottleneck = torch.nn.DataParallel(Bottleneck(self.feature_dim, self.bottleneck_dim))


    def train(self, source_img_set, target_img_seen_set, target_img_unseen_set):
        optimizer = optim.Adam(nn.ModuleList([self.bottleneck]).parameters(), lr=self.learning_rate)
        class_embedding = Variable(torch.from_numpy(self.class_embedding).float(), requires_grad=False).cuda()
        ids_dict = Variable(torch.from_numpy(self.part_ids).float(), requires_grad=False).cuda()
        ids_cvt = self.part_ids.cumsum() - 1
        seen_vec = Variable(torch.from_numpy(self.class_embedding[self.part_ids==1]).float(), requires_grad=False).cuda()
        unseen_vec = Variable(torch.from_numpy(self.class_embedding[self.part_ids==0]).float(), requires_grad=False).cuda()

        loss1 = 0
        loss2 = 0
        
        CE = nn.CrossEntropyLoss().cuda()

        log('train', 'Training Starts')
        for i in range(self.max_iter):

            if i % 1000 == 0 and i != 0:
                self.validate_g(target_img_seen_set, target_img_unseen_set)

            adjust_learning_rate(optimizer, i, self.learning_rate)
            
            source_img, source_label = source_img_set.next_batch(self.batch_size)
            
            source_img = Variable(torch.from_numpy(source_img).float(), requires_grad=False)
            source_label_ce = Variable(torch.from_numpy(ids_cvt[source_label.argmax(-1)]).long(), requires_grad=False).cuda()
            source_label = Variable(torch.from_numpy(source_label).float(), requires_grad=False).cuda()
            
            source_v = torch.mm(source_label, class_embedding)

            source_fc8 = self.bottleneck(source_img)

            source_seen_output = sim(source_fc8, seen_vec.t())
            
            CE_loss = CE(source_seen_output * self.cross_entropy_lambda, source_label_ce)
            loss1 = CE_loss.item()

            if args.entropy_loss_weight > 0:
                source_unseen_output = torch.softmax(torch.mm(source_fc8, unseen_vec.t()) * self.entropy_lambda, dim=1)
                E_loss = (-source_unseen_output * source_unseen_output.log()).sum(1).mean()
                loss2 = E_loss.item()
                CE_loss += args.entropy_loss_weight * E_loss

            optimizer.zero_grad()
            CE_loss.backward()
            optimizer.step()


            if i % self.print_freq == 0:
                print("%05d iter, CrossEntropyLoss: %.4f, EntropyLoss: %.4f" % (i, loss1, loss2))



    def validate(self, target_img_set, part_ids=None):
        self.bottleneck.train(False)

        if part_ids is None:
            part_ids = self.part_ids

        class_embedding = Variable(torch.from_numpy(self.class_embedding).float()).cuda()
        ids_dict = Variable(torch.from_numpy(part_ids).float()).cuda()

        target_img_data, target_img_label = target_img_set.full_data()
        acc = 0.
        acc_g = 0.
        acc_class = np.zeros([self.class_embedding.shape[0]])
        acc_class_g = np.zeros([self.class_embedding.shape[0]])
        n_class = np.zeros([self.class_embedding.shape[0]])

        batch_size = 1000

        for i in range(ceil(target_img_data.shape[0] / batch_size)):
            test_img = target_img_data[i*batch_size:(i+1)*batch_size, :]
            test_img = Variable(torch.from_numpy(test_img).float()).cuda()
            test_label = target_img_label[i*batch_size:(i+1)*batch_size, :]

            rt = lambda x: x.sum(1).view(-1, 1)
            test_fc8 = self.bottleneck(test_img)
            test_ip = sim(test_fc8, class_embedding.t())
            test_output_all = test_ip.exp()
            test_output = test_output_all * (1-ids_dict)

            correct_test_output = test_output.topk(1)[1].squeeze().detach().cpu().numpy() == np.argmax(test_label, 1)
            correct_test_output_all = test_output_all.topk(1)[1].squeeze().detach().cpu().numpy() == np.argmax(test_label, 1)

            acc += correct_test_output.sum()
            acc_class += (correct_test_output[:, None] * test_label).sum(0)
            acc_g += correct_test_output_all.sum()
            acc_class_g += (correct_test_output_all[:, None] * test_label).sum(0)
            n_class += test_label.sum(0)

            if i % 10 == 0:
                log('valid', '%s0000'%(i+1))
        acc /= target_img_data.shape[0]
        acc_g /= target_img_data.shape[0]
        res = []
        res_g = []
        for i in range(self.class_embedding.shape[0]):
            if n_class[i] != 0:
                res.append(acc_class[i] / n_class[i])
                res_g.append(acc_class_g[i] / n_class[i])

        return acc, acc_g, np.mean(res), np.mean(res_g)


    def validate_g(self, seen_set, unseen_set):
        log('valid', 'Validation Start')

        _, _, acc_seen_c, acc_seen_g = self.validate(seen_set, 1 - self.part_ids)
        _, _, acc_unseen_c, acc_unseen_g = self.validate(unseen_set, self.part_ids)

        H = 2 * acc_seen_g * acc_unseen_g / (acc_seen_g + acc_unseen_g)

        print('acc class = %.4f, acc ts = %.4f, acc tr = %.4f, H = %.4f' % 
            (acc_unseen_c * 100, acc_unseen_g * 100, acc_seen_g * 100, H * 100))

        log('valid', 'Validation Finished')



def adjust_learning_rate(optimizer, iter_num, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (1 + 0.01 * iter_num) ** (-0.75)
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr

        
def train():
    dataset = args.dataset
    source_img_tr, target_img_tr, target_img_te, class_embedding, part_ids = GetResClassDatasets(args.data_path+dataset)


    config = dict(
        max_iter = 100000,
        batch_size = args.batch_size,
        learning_rate = args.lr,
        class_embedding = class_embedding,
        class_embedding_dim = class_embedding.shape[1],
        part_ids = part_ids,
        n_class = class_embedding.shape[0],
        entropy_lambda = args.entropy_lambda,
        entropy_loss_weight = args.entropy_loss_weight,
        cross_entropy_lambda = args.cross_entropy_lambda,
    )

    log('param', 'entropy lambda = %.4f, learning rate = %.6f' % (config['entropy_lambda'], config['learning_rate']))
    net = Net(config)
    
    net.train(source_img_tr, target_img_tr, target_img_te)

train()



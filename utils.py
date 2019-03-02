import os
import sys
import numpy as np
import time
import pdb
from datetime import datetime
import scipy.io as sio
from math import ceil
import random

def log(s1, s2):
    print ("%s #%s# %s" % (datetime.now(), s1, s2))


class Normalization(object):
    def __init__(self, data):
        self.n = data.shape[0]
        x1 = data
        self.m = np.sum(x1, 0) / self.n
        x1 = x1 - np.tile(self.m, (self.n, 1))
        self.v = np.sqrt(np.sum(x1**2, 0) / self.n)
        self.v[0] = 1
        x1 = x1 / np.tile(self.v, (self.n, 1))
        self.mx = np.max(x1)

    def transform(self, data):
        self.n = data.shape[0]
        Xn = (data - np.tile(self.m, (self.n, 1))) / np.tile(self.v, (self.n, 1));
        Xn = Xn / self.mx;
        return Xn


def GetResClassDatasets(path):
    class ResDataset(object):
        def __init__(self, data, label, n_class, normalization):
            self.data = normalization.transform(data.T)
            self._n_class = n_class
            self.label = np.array([[1 if i == l else 0 for i in range(n_class)] for l in label])
            self._index = 0
            self._n_samples = self.data.shape[0]
            self._perm = np.arange(self._n_samples)
            np.random.shuffle(self._perm)
        def single_data(self, index):
            return self.data[index, :], self.label[index, :]
        def get_data(self, indices):
            return self.data[indices, :], self.label[indices, :]
        def full_data(self):
            return self.get_data(np.arange(self._n_samples))
        def next_batch(self, batch_size):
            start = self._index
            self._index += batch_size
            if self._index > self._n_samples:
                np.random.shuffle(self._perm)
                start = 0
                self._index = batch_size
            end = self._index
            return self.get_data(self._perm[start:end])
    meta = sio.loadmat(path+'/att_splits.mat')
    feat = sio.loadmat(path+'/res101.mat')
    class_name = [s[0][0] for s in meta['allclasses_names']]
    wordvec = meta['att'].T
    n_class = wordvec.shape[0]
    normalization = Normalization(np.squeeze(feat['features'][:, meta['trainval_loc']-1]).T)
    trainset = ResDataset(np.squeeze(feat['features'][:, meta['trainval_loc']-1]), np.squeeze(feat['labels'][meta['trainval_loc']-1]-1), n_class, normalization)
    test_seen_set = ResDataset(np.squeeze(feat['features'][:, meta['test_seen_loc']-1]), np.squeeze(feat['labels'][meta['test_seen_loc']-1]-1), n_class, normalization)
    test_unseen_set = ResDataset(np.squeeze(feat['features'][:, meta['test_unseen_loc']-1]), np.squeeze(feat['labels'][meta['test_unseen_loc']-1]-1), n_class, normalization)
    part_ids = np.array([1 if i in np.unique(np.squeeze(feat['labels'][meta['trainval_loc']-1]-1)) else 0 for i in range(n_class)])
    return trainset, test_seen_set, test_unseen_set, wordvec, part_ids

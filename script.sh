# To train Zero-shot learning

python train.py --dataset AWA -cel 2

python train.py --dataset SUN -cel 10

python train.py --dataset CUB -cel 8

python train.py --dataset APY -cel 3

# To train Generalized Zero-shot learning

python train.py --dataset AWA -cel 2 -el .03 -elw 1

python train.py --dataset SUN -cel 5 -el .1 -elw 1

python train.py --dataset CUB -cel 5 -el .03 -elw 1

python train.py --dataset APY -cel 3 -el 1. -elw 0
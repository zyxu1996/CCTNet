#!/bin/sh
################### Test #################

#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29505 test.py --dataset barley --val_batchsize 8 --models cctnet --head seghead --crop_size 512 512 --trans_cnn cswin_tiny resnet50 --save_dir work_dir --base_dir ../../ --information num1


################### Train #################

# barley
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29506 train.py --dataset barley --end_epoch 50 --lr 0.0001 --train_batchsize 4 --models cctnet --head seghead --crop_size 512 512 --trans_cnn cswin_tiny resnet50 --use_mixup 0 --information num1

# vaihingen
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29507 train.py --dataset vaihingen --end_epoch 100 --lr 0.0003 --train_batchsize 4 --models cctnet --head seghead --crop_size 512 512 --trans_cnn cswin_tiny resnet50 --use_mixup 0 --information num2


# potsdam
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29508 train.py --dataset potsdam --end_epoch 50 --lr 0.0001 --train_batchsize 4 --models cctnet --head seghead --crop_size 512 512 --trans_cnn cswin_tiny resnet50 --use_mixup 0 --information num3
import os
import argparse
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import numpy as np
import math

from dataset import RemoteData, label_to_RGB, RGB_to_label
from seg_metric import SegmentationMetric
import cv2
from mutil_scale_test import MultiEvalModule
import logging
import warnings
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument("--dataset", type=str, default='vaihingen', choices=['potsdam', 'vaihingen', 'barley'])
    parser.add_argument("--val_batchsize", type=int, default=16)
    parser.add_argument("--crop_size", type=int, nargs='+', default=[512, 512], help='H, W')
    parser.add_argument("--models", type=str, default='danet',
                        choices=['danet', 'bisenetv2', 'pspnet', 'segbase', 'swinT', 'deeplabv3',
                                 'fcn', 'fpn', 'unet', 'resT', 'cctnet', 'beit', 'cswin', 'volo', 'transformer'])
    parser.add_argument("--head", type=str, default='uperhead')
    parser.add_argument("--trans_cnn", type=str, nargs='+', default=['cswin_tiny', 'resnet50'], help='ttansformer, cnn')
    parser.add_argument("--use_edge", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default='work_dir')
    parser.add_argument("--base_dir", type=str, default='./')
    parser.add_argument("--information", type=str, default='RS')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--save_gpu_memory", type=int, default=0)
    parser.add_argument("--val_full_img", type=int, default=1)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args2 = parser.parse_args()
    return args2


class FullModel(nn.Module):

    def __init__(self, model):
        super(FullModel, self).__init__()
        self.model = model

    def forward(self, input):

        return self.model(input)


def get_model():
    models = args2.models
    if models in ['swinT', 'resT', 'beit', 'cswin', 'volo']:
        print(models, args2.head)
    elif models in ['transformer', 'cctnet']:
        print(models, args2.trans_cnn, args2.head)
    else:
        print(models)
    if args2.dataset in ['potsdam', 'vaihingen']:
        nclass = 6
    if args2.dataset in ['barley']:
        nclass = 4
    assert models in ['danet', 'bisenetv2', 'pspnet', 'segbase', 'swinT', 'deeplabv3',
                      'fcn', 'fpn', 'unet', 'resT', 'cctnet', 'beit', 'cswin', 'volo', 'transformer']
    if models == 'danet':
        from models.danet import DANet
        model = DANet(nclass=nclass, backbone='resnet50', pretrained_base=False)
    if models == 'bisenetv2':
        from models.bisenetv2 import BiSeNetV2
        model = BiSeNetV2(nclass=nclass)
    if models == 'pspnet':
        from models.pspnet import PSPNet
        model = PSPNet(nclass=nclass, backbone='resnet50', pretrained_base=False)
    if models == 'segbase':
        from models.segbase import SegBase
        model = SegBase(nclass=nclass, backbone='resnet50', pretrained_base=False)
    if models == 'swinT':
        from models.swinT import swin_large as swinT
        model = swinT(nclass=nclass, pretrained=False, aux=True, head=args2.head, edge_aux=args2.use_edge)
    if models == 'resT':
        from models.resT import rest_large as resT
        model = resT(nclass=nclass, pretrained=False, aux=True, head=args2.head, edge_aux=args2.use_edge)
    if models == 'deeplabv3':
        from models.deeplabv3 import DeepLabV3
        model = DeepLabV3(nclass=nclass, backbone='resnet50', pretrained_base=False)
    if models == 'fcn':
        from models.fcn import FCN16s
        model = FCN16s(nclass=nclass)
    if models == 'fpn':
        from models.fpn import FPN
        model = FPN(nclass=nclass)
    if models == 'unet':
        from models.unet import UNet
        model = UNet(nclass=nclass)
    if models == 'cctnet':
        from models.cctnet import CCTNet
        model = CCTNet(transformer_name=args2.trans_cnn[0], cnn_name=args2.trans_cnn[1], nclass=nclass, img_size=args2.crop_size[0],
                       pretrained=False, aux=True, head=args2.head, edge_aux=args2.use_edge)
    if models == 'beit':
        from models.beit import beit_base as beit
        model = beit(nclass=nclass, img_size=args2.crop_size[0], pretrained=False, aux=True, head=args2.head, edge_aux=args2.use_edge)
    if models == 'cswin':
        from models.cswin import cswin_tiny as cswin
        model = cswin(nclass=nclass, img_size=args2.crop_size[0], pretrained=False, aux=True, head=args2.head, edge_aux=args2.use_edge)
    if models == 'volo':
        from models.volo import volo_d4 as volo
        model = volo(nclass=nclass, img_size=args2.crop_size[0], pretrained=False, aux=True, head=args2.head, edge_aux=args2.use_edge)
    if models == 'banet':
        from models.banet import BANet
        model = BANet(nclass=nclass)
    if models == 'transformer':
        from  models.transformer import Transformer
        model = Transformer(transformer_name=args2.trans_cnn[0], nclass=nclass, img_size=args2.crop_size[0],
                            pretrained=False, aux=True, head=args2.head, edge_aux=args2.use_edge)

    model = FullModel(model)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args2.local_rank], output_device=args2.local_rank, find_unused_parameters=True)
    return model


args2 = parse_args()

cudnn.benchmark = True
cudnn.deterministic = False
cudnn.enabled = True
distributed = True

device = torch.device(('cuda:{}').format(args2.local_rank))


if distributed:
    torch.cuda.set_device(args2.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://",
    )

data_dir = os.path.join(args2.base_dir, 'data')

if not args2.val_full_img:
    remotedata_val = RemoteData(base_dir=data_dir, train=False,
                          dataset=args2.dataset, crop_szie=args2.crop_size)
    if distributed:
        val_sampler = DistributedSampler(remotedata_val)
    else:
        val_sampler = None
    dataloader_val = DataLoader(
        remotedata_val,
        batch_size=args2.val_batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler)
else:
    remotedata_val_full = RemoteData(base_dir=data_dir, train=False,
                               dataset=args2.dataset, crop_szie=args2.crop_size, val_full_img=True)
    if distributed:
        full_val_sampler = DistributedSampler(remotedata_val_full)
    else:
        full_val_sampler = None
    dataloader_val_full = DataLoader(
        remotedata_val_full,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=full_val_sampler)


def val(model, weight_path):

    if args2.dataset in ['potsdam', 'vaihingen']:
        nclasses = 6
    if args2.dataset in ['barley']:
        nclasses = 4
    model.eval()
    metric = SegmentationMetric(numClass=nclasses)
    with torch.no_grad():
        model_state_file = weight_path
        if os.path.isfile(model_state_file):
            print('loading checkpoint successfully')
            logging.info("=> loading checkpoint '{}'".format(model_state_file))
            checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            # checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
        else:
            warnings.warn('weight is not existed !!!"')

        for i, sample in enumerate(dataloader_val):

            images, labels = sample['image'], sample['label']
            if args2.dataset in ['barley']:
                alphas = sample['alpha']
            images = images.cuda()
            labels = labels.long().squeeze(1)
            logits = model(images)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            print("test:{}/{}".format(i, len(dataloader_val)))
            logits = logits.argmax(dim=1)
            logits = logits.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            if args2.dataset in ['barley']:
                metric.addBatch(logits[alphas > 0], labels[alphas > 0])
            else:
                metric.addBatch(logits, labels)
        result_count(metric)


def mutil_scale_val(model, weight_path, object_path):
    if args2.dataset in ['potsdam', 'vaihingen']:
        nclasses = 6
    if args2.dataset in ['barley']:
        nclasses = 4
    model.eval()
    metric = SegmentationMetric(nclasses)
    with torch.no_grad():
        model_state_file = weight_path
        if os.path.isfile(model_state_file):
            print('loading checkpoint successfully')
            logging.info("=> loading checkpoint '{}'".format(model_state_file))
            checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            elif 'model' in checkpoint:
                checkpoint = checkpoint['model']
            else:
                checkpoint = checkpoint
            checkpoint = {k: v for k, v in checkpoint.items() if not 'n_averaged' in k}
            # checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
        else:
            warnings.warn('weight is not existed !!!"')
        model = MultiEvalModule(model, nclass=nclasses, flip=False, scales=[1.0], save_gpu_memory=args2.save_gpu_memory,
                                crop_size=args2.crop_size, stride_rate=1, get_batch=args2.val_batchsize)
        for i, sample in enumerate(dataloader_val_full):

            images, labels, names = sample['image'], sample['label'], sample['name']
            if args2.dataset in ['barley']:
                alphas = sample['alpha']
            images = images.cuda()
            labels = labels.long().squeeze(1)
            logits = model(images)
            print("test:{}/{}".format(i, len(dataloader_val_full)))
            logits = logits.argmax(dim=1)
            logits = logits.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            if args2.dataset in ['barley']:
                metric.addBatch(logits[alphas > 0], labels[alphas > 0])
            else:
                metric.addBatch(logits, labels)
            vis_logits = label_to_RGB(logits.squeeze(), classes=nclasses)[:, :, ::-1]
            save_path = os.path.join(object_path, 'outputs', names[0] + '.png')
            cv2.imwrite(save_path, vis_logits)
        result_count(metric)


def result_count(metric):
    iou = metric.IntersectionOverUnion()
    acc = metric.Accuracy()
    f1 = metric.F1()
    precision = metric.Precision()
    recall = metric.Recall()
    if args2.dataset in ['potsdam', 'vaihingen']:
        iou, f1, precision, recall = iou[0:5], f1[0:5], precision[0:5], recall[0:5]  # ignore background
    miou = np.nanmean(iou)
    mf1 = np.nanmean(f1)
    mprecision = np.nanmean(precision)
    mrecall = np.nanmean(recall)

    iou = reduce_tensor(torch.from_numpy(np.array(iou)).to(device) / get_world_size()).cpu().numpy()
    miou = reduce_tensor(torch.from_numpy(np.array(miou)).to(device) / get_world_size()).cpu().numpy()
    acc = reduce_tensor(torch.from_numpy(np.array(acc)).to(device) / get_world_size()).cpu().numpy()
    f1 = reduce_tensor(torch.from_numpy(np.array(f1)).to(device) / get_world_size()).cpu().numpy()
    mf1 = reduce_tensor(torch.from_numpy(np.array(mf1)).to(device) / get_world_size()).cpu().numpy()
    precision = reduce_tensor(torch.from_numpy(np.array(precision)).to(device) / get_world_size()).cpu().numpy()
    mprecision = reduce_tensor(torch.from_numpy(np.array(mprecision)).to(device) / get_world_size()).cpu().numpy()
    recall = reduce_tensor(torch.from_numpy(np.array(recall)).to(device) / get_world_size()).cpu().numpy()
    mrecall = reduce_tensor(torch.from_numpy(np.array(mrecall)).to(device) / get_world_size()).cpu().numpy()

    if args2.local_rank == 0:
        print('\n')
        logging.info('####################### full image val ###########################')
        print('|{}:{}{}{}{}|'.format(str('CLASSES').ljust(24),
                                     str('Precision').rjust(10), str('Recall').rjust(10),
                                     str('F1').rjust(10), str('IOU').rjust(10)))
        logging.info('|{}:{}{}{}{}|'.format(str('CLASSES').ljust(24),
                                            str('Precision').rjust(10), str('Recall').rjust(10),
                                            str('F1').rjust(10), str('IOU').rjust(10)))
        for i in range(len(iou)):
            print('|{}:{}{}{}{}|'.format(str(CLASSES[i]).ljust(24),
                                         str(round(precision[i], 4)).rjust(10), str(round(recall[i], 4)).rjust(10),
                                         str(round(f1[i], 4)).rjust(10), str(round(iou[i], 4)).rjust(10)))
            logging.info('|{}:{}{}{}{}|'.format(str(CLASSES[i]).ljust(24),
                                                str(round(precision[i], 4)).rjust(10),
                                                str(round(recall[i], 4)).rjust(10),
                                                str(round(f1[i], 4)).rjust(10), str(round(iou[i], 4)).rjust(10)))
        print('mIoU:{} ACC:{} mF1:{} mPrecision:{} mRecall:{}'.format(round(miou * 100, 2),
                                                                      round(acc * 100, 2), round(mf1 * 100, 2),
                                                                      round(mprecision * 100, 2),
                                                                      round(mrecall * 100, 2)))
        logging.info('mIoU:{} ACC:{} mF1:{} mPrecision:{} mRecall:{}'.format(round(miou * 100, 2),
                                                                             round(acc * 100, 2), round(mf1 * 100, 2),
                                                                             round(mprecision * 100, 2),
                                                                             round(mrecall * 100, 2)))
        print('\n')


def restore_part_img(oh=6000, ow=6000, overlap=1024, args2=None, test_list=['image_2']):
    """'image_2':num_h=10, num_w=16, 'image_1':num_h=10, num_w=10"""
    for filename in test_list:
        if filename == 'image_1':
            h, w = 50141, 47161
        if filename == 'image_2':
            h, w = 46050, 77470
        num_h, num_w = math.ceil((h - oh) / (oh - overlap)) + 1, math.ceil((w - ow) / (ow - overlap)) + 1
        for i in range(num_w):
            for j in range(num_h):
                labels_view_dir = os.path.join(args2.base_dir, f'./data/barley/labels_view/{filename}_{j}_{i}.png')
                if os.path.exists(f'./outputs/{filename}_{j}_{i}.png'):
                    part_img = cv2.imread(f'./outputs/{filename}_{j}_{i}.png')
                elif os.path.exists(labels_view_dir):
                    part_img = cv2.imread(labels_view_dir)
                else:
                    part_img = np.ones(shape=(oh, ow, 3), dtype=np.uint8) * 255
                if j == 0:
                    w_patch = part_img[0:oh - overlap // 2, ...]
                elif j < num_h - 1:
                    w_patch = np.concatenate((w_patch, part_img[overlap // 2:oh - overlap // 2, ...]), 0)
                else:
                    end_h = w_patch.shape[0]
                    w_patch = np.concatenate((w_patch, part_img[oh - (h - end_h):oh, ...]), 0)
            if i == 0:
                h_patch = w_patch[:, 0:ow - overlap // 2, :]
            elif i < num_w - 1:
                h_patch = np.concatenate((h_patch, w_patch[:, overlap // 2:ow - overlap // 2, :]), 1)
            else:
                end_w = h_patch.shape[1]
                h_patch = np.concatenate((h_patch, w_patch[:, ow - (w - end_w):ow, :]), 1)
        print(h_patch.shape)
        cv2.imwrite(f'./outputs/{filename}.png', h_patch)
        img_resize = cv2.resize(h_patch, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f'./outputs/{filename}_size0.1.png', img_resize)
        return h_patch[..., ::-1]  # BGR TO RGB


def val_full_label(args2, predict=None, test_file='image_2', save_memory=False):
    label_dir = os.path.join(args2.base_dir, 'data/barley/labels_complete', f'{test_file}.png')
    label = np.array(Image.open(label_dir))

    if predict is None:
        predict_dir = f'./outputs/{test_file}.png'
        predict = np.array(Image.open(predict_dir))
    predict = RGB_to_label(predict, classes=4)

    alpha_dir = os.path.join(args2.base_dir, 'data/barley/alphas_complete', f'{test_file}.png')
    alpha = np.array(Image.open(alpha_dir))

    img_save = Image.fromarray(predict)
    img_save.save(f'./outputs/{test_file}_mask.png')

    if save_memory:
        predict = cv2.resize(predict, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)
        alpha = cv2.resize(alpha, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)

    logging.info(f'######## complete image test of {test_file} and ignore the transparent area !!! ########')
    metric = SegmentationMetric(numClass=4)
    metric.addBatch(predict[alpha > 0], label[alpha > 0])
    result_count(metric)


def get_model_path(args2):
    object_path, weight_path = None, None
    file_dir = os.path.join(args2.base_dir, args2.save_dir)
    file_list = os.listdir(file_dir)
    for file in file_list:
        if args2.models in file and args2.information in file:
            weight_path = os.path.join(file_dir, file, 'weights', 'best_weight.pkl')
            object_path = os.path.join(file_dir, file)
    if object_path is None or weight_path is None:
        tmp_path = os.path.join(file_dir, 'tmp_save')
        output_path = os.path.join(tmp_path, 'outputs')
        weight_path = os.path.join(tmp_path, 'weights')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)
        object_path = tmp_path
        weight_path = weight_path + '/best_weight.pkl'
        warnings.warn('path is not defined, will be set as "./work_dir/tmp_save"')
    return object_path, weight_path


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # os.environ.setdefault('RANK', '0')
    # os.environ.setdefault('WORLD_SIZE', '1')
    # os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    # os.environ.setdefault('MASTER_PORT', '29555')

    object_path, weight_path = get_model_path(args2)
    save_log = os.path.join(object_path, 'test.log')
    logging.basicConfig(filename=save_log, level=logging.INFO)

    if args2.dataset in ['potsdam', 'vaihingen']:
        CLASSES = ('Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car', 'Clutter/background')
    if args2.dataset in ['barley']:
        CLASSES = ('Background', 'Cured tobacco', 'Corn', 'Barley rice')
    model = get_model()
    if args2.val_full_img:
        mutil_scale_val(model, weight_path, object_path)
        torch.cuda.synchronize()  # synchronize all gpus
        if args2.dataset in ['barley'] and get_rank() == 0:
            predict_rgb = None
            predict_rgb1 = restore_part_img(args2=args2, test_list=['image_1'], overlap=0)
            predict_rgb2 = restore_part_img(args2=args2, test_list=['image_2'], overlap=0)
            val_full_label(args2=args2, predict=predict_rgb1, test_file='image_1', save_memory=True)
            val_full_label(args2=args2, predict=predict_rgb2, test_file='image_2', save_memory=True)
    else:
        val(model, weight_path)







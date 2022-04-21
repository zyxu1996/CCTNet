from PIL import Image
import os
import numpy as np
from PIL import ImageFile
import math
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def label_to_RGB(image, classes=4):
    RGB = np.zeros(shape=[image.shape[0], image.shape[1], 3], dtype=np.uint8)
    if classes == 6:
        palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    if classes == 4:
        palette = [[255, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    for i in range(classes):
        index = image == i
        RGB[index] = np.array(palette[i])
    return RGB


def divide_img(image, oh, ow, filename, save_dir, write_txt_dir=None):
    """No overlap, and the last square maybe small than oh ow"""
    if write_txt_dir is not None:
        txt = open(write_txt_dir, 'w')
    h, w = image.shape[0:2]
    num_h, num_w = h // oh + 1, w // ow + 1
    for i in range(num_w):
        for j in range(num_h):
            h1 = min((j + 1) * oh, h)
            w1 = min((i + 1) * ow, w)
            if len(image.shape) == 2:
                image_part = image[j * oh:h1, i * ow:w1]
            else:
                image_part = image[j * oh:h1, i * ow:w1, :]
            image_part = Image.fromarray(image_part)
            image_part.save(os.path.join(save_dir, f'{filename}_{j}_{i}.png'))  # j:h, i:w
            if write_txt_dir is not None:
                txt.write(f'{filename}_{j}_{i}' + '\n')


def divide_img_overlap(image, oh, ow, filename, save_dir, write_txt_dir=None, overlap=1024):
    """Divide img with an overlap, the last square is back trace to oh ow"""
    if write_txt_dir is not None:
        txt = open(write_txt_dir, 'w')
        path, name = os.path.split(write_txt_dir)
        txt_clean = open(os.path.join(path, os.path.splitext(name)[0] + '_clean.txt'), 'w')
    h, w = image.shape[0:2]
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    num_h, num_w = math.ceil((h - oh) / (oh - overlap)) + 1, math.ceil((w - ow) / (ow - overlap)) + 1
    for i in range(num_w):
        for j in range(num_h):
            if i < num_w - 1 and j < num_h - 1:
                image_part = image[(oh - overlap) * j:(oh - overlap) * j + oh, (ow - overlap) * i:(ow - overlap) * i + ow, :]
            if i < num_w - 1 and j == num_h - 1:
                image_part = image[h - oh:h, (ow - overlap) * i:(ow - overlap) * i + ow, :]
            if i == num_w - 1 and j < num_h - 1:
                image_part = image[(oh - overlap) * j:(oh - overlap) * j + oh, w - ow:w, :]
            if i == num_w - 1 and j == num_h - 1:
                image_part = image[h - oh:h, w - ow:w, :]
            image_part = image_part.squeeze()
            if write_txt_dir is not None:
                if np.any(image_part[..., 3] > 0):
                    txt_clean.write(f'{filename}_{j}_{i}' + '\n')
                txt.write(f'{filename}_{j}_{i}' + '\n')
            image_part = Image.fromarray(image_part)
            image_part.save(os.path.join(save_dir, f'{filename}_{j}_{i}.png'))  # j:h, i:w


def restore_part_img(oh=6000, ow=6000, overlap=1024, filename='image_1'):
    """restore patches of the image, the last square is back traced to oh ow"""
    root = '/data/xzy/datasets/'
    dataset = f'barley_hw6000_s{overlap}'
    if filename == 'image_1':
        h, w = 50141, 47161
    if filename == 'image_2':
        h, w = 46050, 77470
    num_h, num_w = math.ceil((h - oh) / (oh - overlap)) + 1, math.ceil((w - ow) / (ow - overlap)) + 1
    for i in range(num_w):
        for j in range(num_h):
            part_img = Image.open(os.path.join(root, dataset, f'images/{filename}_{j}_{i}.png'))
            part_img = np.array(part_img)
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
    h_patch = cv2.resize(h_patch, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)
    full_img = Image.fromarray(h_patch)
    full_img.save(os.path.join(root, dataset, f'{filename}.png'))


def apply_divide_img_label(overlap=1024):
    """read images and divide them with an overlap"""
    root = '/data/zyxu/dataset/barley/'
    save = f'barley_hw6000_s{overlap}'
    img_dir = 'images_complete/'
    label_dir = 'labels_complete/'
    img_save_dir = os.path.join(root, f'{save}/images/')
    label_save_dir = os.path.join(root, f'{save}/labels/')
    file = [f'image_{1}', f'image_{2}']

    for filename in file:
        if filename == f'image_{1}':
            txt_name = os.path.join(root, f'{save}/annotations/train_full.txt')
        if filename == f'image_{2}':
            txt_name = os.path.join(root, f'{save}/annotations/test_full.txt')

        img = Image.open(os.path.join(root, img_dir, filename + '.png'))
        img = np.array(img)
        divide_img_overlap(img, 6000, 6000, filename, img_save_dir, txt_name, overlap=overlap)

        label = Image.open(os.path.join(root, label_dir, filename + '.png'))
        label = np.array(label)
        divide_img_overlap(label, 6000, 6000, filename, label_save_dir, overlap=overlap)


def clean_white_background():
    """remove transparent patches from the train and test txt """
    img_dir = './data/barley/images/'
    img_list = os.listdir(img_dir)
    train_no_alpha = open('./data/barley/annotations/train_no_alpha.txt', 'w')
    test_no_alpha = open('./data/barley/annotations/test_no_alpha.txt', 'w')
    for file in img_list:
        file = file.strip()
        img = Image.open(os.path.join(img_dir, file))
        img = np.array(img)
        alpha = img[..., 3]
        if np.any(alpha > 0):
            if 'image_1' in file:
                train_no_alpha.write(file[:-4] + '\n')
            if 'image_2' in file:
                test_no_alpha.write(file[:-4] + '\n')


def count_nums():
    """count pixels of each classes in images"""
    label_train_dir = '/data/xzy/datasets/barley/labels_full/image_1_label.png'
    label_test_dir = '/data/xzy/datasets/barley/labels_full/image_2_label.png'
    label_train = np.array(Image.open(label_train_dir))
    label_test = np.array(Image.open(label_test_dir))
    h0, w0, h1, w1 = label_train.shape[0], label_train.shape[1], label_test.shape[0], label_test.shape[1]
    for i in range(4):
        print('train pixel{}:{:.6f}'.format(i, np.sum(label_train == i) / (h0 * w0)))
        # 0.870185 0.066412 0.006110 0.057294
        # class012: [0.51158563 0.04706662 0.44134775]
    for i in range(4):
        print('test pixel{}:{:.6f}'.format(i, np.sum(label_test == i) / (h1 * w1)))
        # 0.926607 0.002005 0.033732 0.037655
        # class012: [0.02731905 0.45961413 0.51306682]


def rearrange_dataset(oh=6000, ow=6000, overlap=1024):
    """fuse image_1 and image_2 to generate new train and test file"""
    root = '/data/zyxu/dataset/barley/'
    dataset = f'barley_hw6000_s{overlap}'
    train_txt = open(os.path.join(root, dataset, f'annotations/train.txt'), 'w')
    test_txt = open(os.path.join(root, dataset, f'annotations/test.txt'), 'w')
    for filename in ['image_1', 'image_2']:
        if filename == 'image_1':
            h, w = 50141, 47161
        if filename == 'image_2':
            h, w = 46050, 77470
        num_h, num_w = math.ceil((h - oh) / (oh - overlap)) + 1, math.ceil((w - ow) / (ow - overlap)) + 1
        for i in range(num_w):
            for j in range(num_h):
                part_img = Image.open(os.path.join(root, dataset, f'images/{filename}_{j}_{i}.png'))
                part_img = np.array(part_img)
                if (i + j) % 2 == 0 and np.any(part_img[..., 3] > 0):
                    train_txt.write(f'{filename}_{j}_{i}' + '\n')
                if (i + j) % 2 == 1 and np.any(part_img[..., 3] > 0):
                    test_txt.write(f'{filename}_{j}_{i}' + '\n')


def resize():
    """resize image to 1/10"""
    filename = 'image_2'
    img1_dir = f'./data/barley/labels_view/{filename}.png'
    img1 = np.array(Image.open(img1_dir))
    img = cv2.resize(img1, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)
    img = Image.fromarray(img)
    img.save(f'./data/barley/images_size0.1/{filename}_labels_view.png')


def label_view():
    """view label to rgb"""
    root = '/data/xzy/datasets/barley/'
    dataset = f''
    train_txt = os.path.join(root, dataset, 'annotations/train.txt')
    test_txt = os.path.join(root, dataset, 'annotations/test.txt')
    file = open(train_txt, 'r').readlines() + open(test_txt, 'r').readlines()
    for name in file:
        name = name.strip()
        label = np.array(Image.open(os.path.join(root, dataset, 'labels', name + '.png')))
        label = label_to_RGB(label, 4)
        label = Image.fromarray(label)
        label.save(os.path.join(root, dataset, f'labels_view/{name}.png'))


def get_alpha():
    """get alpha channel and save it"""
    name = 'image_2.png'
    image_dir = './data/barley/images_complete/'
    image = np.array(Image.open(os.path.join(image_dir, name)))
    alpha = image[..., 3]
    alpha = Image.fromarray(alpha)
    alpha.save(f'./data/barley/alphas_complete/{name}')


if __name__ == '__main__':
    # apply_divide_img_label()
    # restore_part_img()

    # label1 = np.array(Image.open('./data/barley/barley_hw6000_s1024/image_2.png'))
    # label2 = np.array(Image.open('./data/barley/labels_complete/image_2.png'))
    # print(label1.shape, label2.shape)
    # print(np.all(label1 == label2))
    # print(np.sum(label1), np.sum(label2))

    # apply_divide_img_label(overlap=0)
    # restore_part_img(overlap=0)
    # rearrange_dataset(overlap=0)
    label_view()




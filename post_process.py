'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description :
LastEditTime: 2020-11-27 03:42:46
'''
import os
import threading
import cv2 as cv
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects
from argparse import ArgumentParser
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class MyThread(threading.Thread):

    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None


def label_resize_vis(label, img=None, alpha=0.5):
    '''
    :param label:原始标签
    :param img: 原始图像
    :param alpha: 透明度
    :return: 可视化标签
    '''
    def label_to_RGB(image, classes=6):
        RGB = np.zeros(shape=[image.shape[0], image.shape[1], 3], dtype=np.uint8)
        if classes == 6:  # potsdam and vaihingen
            palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
        if classes == 4:  # barley
            palette = [[255, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
        for i in range(classes):
            index = image == i
            RGB[index] = np.array(palette[i])
        return RGB

    # label = cv.resize(label.copy(), None, fx=0.1, fy=0.1)
    anno_vis = label_to_RGB(label, classes=4)
    if img is None:
        return anno_vis
    else:
        overlapping = cv.addWeighted(img, alpha, anno_vis, 1 - alpha, 0)
        return overlapping


def remove_small_objects_and_holes(class_type, label, min_size, area_threshold, in_place=True):
    print("------------- class_n : {} start ------------".format(class_type))
    if class_type == 3:
        # kernel = cv.getStructuringElement(cv.MORPH_RECT,(500,500))
        # label = cv.dilate(label,kernel)
        # kernel = cv.getStructuringElement(cv.MORPH_RECT,(10,10))
        # label = cv.erode(label,kernel)
        label = remove_small_objects(label == 1, min_size=min_size, connectivity=1, in_place=in_place)
        label = remove_small_holes(label == 1, area_threshold=area_threshold, connectivity=1, in_place=in_place)
    else:
        label = remove_small_objects(label == 1, min_size=min_size, connectivity=1, in_place=in_place)
        label = remove_small_holes(label == 1, area_threshold=area_threshold, connectivity=1, in_place=in_place)
    print("------------- class_n : {} finished ------------".format(class_type))
    return label


def RGB_to_label(image=None, classes=6):
    if classes == 6:  # potsdam and vaihingen
        palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    if classes == 4:  # barley
        palette = [[255, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    label = np.zeros(shape=[image.shape[0], image.shape[1]], dtype=np.uint8)
    for i in range(len(palette)):
        index = image == np.array(palette[i])
        index[..., 0][index[..., 1] == False] = False
        index[..., 0][index[..., 2] == False] = False
        label[index[..., 0]] = i
    return label



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_n", type=int, default=2, help="传入1或2，指定")
    parser.add_argument("--image_path", type=str, default='./outputs', help="传入image_n_predict所在路径")
    parser.add_argument("--threshold", type=int, default=2000)
    arg = parser.parse_args()
    image_n = arg.image_n
    image_path = arg.image_path
    threshold = arg.threshold

    if image_n == 1:
        source_image = cv.imread("../../data/barley/images_size0.1/image_1.png")
    elif image_n == 2:
        source_image = cv.imread("../../data/barley/images_size0.1/image_2.png")
    else:
        raise ValueError("image_n should be 1 or 2, Got {} ".format(image_n))

    img_mask_dir = os.path.join(image_path, f'image_{image_n}_mask.png')
    img_dir = os.path.join(image_path, f'image_{image_n}.png')
    if os.path.exists(img_mask_dir):
        image = np.asarray(Image.open(img_mask_dir))
    elif os.path.exists(img_dir):
        image = np.asarray(Image.open(img_dir))
    else:
        raise ValueError(f"Not found image_{image_n}_mask.png or image_{image_n}.png")

    if len(image.shape) == 3:
        image = RGB_to_label(image, classes=4)
        image_save = Image.fromarray(image)
        image_save.save(os.path.join(image_path, f'image_{image_n}_mask.png'))

    image = cv.resize(image, None, fx=0.1, fy=0.1, interpolation=cv.INTER_NEAREST)  # because over memory

    label = to_categorical(image, num_classes=4, dtype='uint8')

    threading_list = []
    for i in range(4):
        t = MyThread(remove_small_objects_and_holes, args=(i, label[:, :, i], threshold, threshold, True))
        threading_list.append(t)
        t.start()

    # 等待所有线程运行完毕
    result = []
    for t in threading_list:
        t.join()
        result.append(t.get_result()[:, :, None])

    label = np.concatenate(result, axis=2)

    label = np.argmax(label, axis=2).astype(np.uint8)
    cv.imwrite('./outputs/image_' + str(image_n) + "_predict.png", label)
    mask = label_resize_vis(label, source_image)
    cv.imwrite('./outputs/vis_image_' + str(image_n) + "_predict.jpg", mask[..., ::-1])
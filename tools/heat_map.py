import cv2
import time
import os
import matplotlib.pyplot as plt
import numpy as np


save_path='./heatmap/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

def draw_features(x,savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    b, c, h, w = x.shape
    for i in range(int(c)):
        plt.subplot(h, w, i + 1)
        plt.axis('off')
        img = x[0, i, :, :].cpu().numpy()
        print('img_shape', img.shape)
        # print('img', img)
        # print(width,height)
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255
        img=img.astype(np.uint8)   #转成unit8
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map
        # img = img[:, :, ::-1]   #注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i, c))
        print(img.shape)
        img = cv2.resize(img, (768, 768), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(save_path + savename + str(i) + '.png', img)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))


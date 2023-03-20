import numpy as np
import cv2
import os
import struct
 
def trans(image, label, save):#image位置，label位置和转换后的数据保存位置
    if 'train' in os.path.basename(image):
        prefix = 'train'
    else:
        prefix = 'test'
 
    labelIndex = 0
    imageIndex = 0
    i = 0
    lbdata = open(label, 'rb').read()
    magic, nums = struct.unpack_from(">II", lbdata, labelIndex)
    labelIndex += struct.calcsize('>II')
 
    imgdata = open(image, "rb").read()
    magic, nums, numRows, numColumns = struct.unpack_from('>IIII', imgdata, imageIndex)
    imageIndex += struct.calcsize('>IIII')
 
    for i in range(nums):
        label = struct.unpack_from('>B', lbdata, labelIndex)[0]
        labelIndex += struct.calcsize('>B')
        im = struct.unpack_from('>784B', imgdata, imageIndex)
        imageIndex += struct.calcsize('>784B')
        im = np.array(im, dtype='uint8')
        img = im.reshape(28, 28)
        save_name = os.path.join(save, '{}_{}_{}.jpg'.format(prefix, i, label))
        cv2.imwrite(save_name, img)
 
if __name__ == '__main__':
    #需要更改的文件路径！！！！！！
    #此处是原始数据集位置
    train_images = './data/train-images.idx3-ubyte'
    train_labels = './data/train-labels.idx1-ubyte'
    test_images ='./data/t10k-images.idx3-ubyte'
    test_labels = './data/t10k-labels.idx1-ubyte'
    #此处是我们将转化后的数据集保存的位置
    save_train ='./MNIST_data/train_images/'
    save_test ='./MNIST_data/test_images/'
    
    if not os.path.exists(save_train):
        os.makedirs(save_train)
    if not os.path.exists(save_test):
        os.makedirs(save_test)
 
    trans(test_images, test_labels, save_test)
    trans(train_images, train_labels, save_train)
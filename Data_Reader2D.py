import os
from os import listdir
import cv2
import numpy as np
import random
import tensorflow as tf


class Data_Reader(object):
    def __init__(self, ImageDir, labelDir, BatchSize=1, num_classes = 2):# numpoints = 6
        self.ImageDir = ImageDir
        self.LabelDir = labelDir
        self.BatchSize = BatchSize
        self.start_index = 0
        self.num_classes = num_classes
        self.image_dir = listdir(self.ImageDir)
        self.label_dir = listdir(self.LabelDir)
        random.shuffle(self.image_dir)
        print ("total image: ", len(self.image_dir))


    def getBatch(self):
        """
        img = np.zeros((self.BatchSize,256,256,1))
        label = np.zeros((self.BatchSize,256,256))
        ind = 0
        """
        image = np.zeros((self.BatchSize, 256, 256, 1))
        for ind in range(self.BatchSize):
            Img = cv2.imread(self.ImageDir + self.image_dir[self.start_index + ind], 0)
            Img = cv2.resize(Img, (256, 256))
            image[ind,:,:,0] = Img

        label = np.zeros((self.BatchSize, 256, 256, 3))
        for ind in range(self.BatchSize):
            lab = cv2.imread(self.LabelDir + self.image_dir[self.start_index + ind][:-4] + '.png', 0)
            lab1 = (lab == 1).astype(np.uint8)
            lab1 = cv2.resize(lab1, (256, 256))

            contours, _ = cv2.findContours(lab1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
            canv1 = np.zeros(lab1.shape).astype(np.uint8)
            cv2.drawContours(canv1, contours, -1, 1, -1)

            lab2 = (lab == 2).astype(np.uint8)
            lab2 = cv2.resize(lab2, (256, 256))

            contours, _ = cv2.findContours(lab2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
            canv2 = np.zeros(lab2.shape).astype(np.uint8)
            cv2.drawContours(canv2, contours, -1, 1, -1)
            #print(set(lab.ravel()))
            label[ind, :, :, 1] = canv1#(lab == 1).astype(np.uint8)
            label[ind, :, :, 2] = canv2#(lab == 2).astype(np.uint8)

            temp = np.ones((256, 256)).astype(np.uint8)
            temp[(canv1 + canv2) > 0] = 0
            label[ind, :, :, 0] = temp

        self.start_index += self.BatchSize
        if self.start_index + self.BatchSize >= len(self.image_dir):
            self.start_index = 0
            print("######################################################")
            print ("epoch finished 1")
            print("######################################################")
            random.shuffle(self.image_dir)
        return image, label


data_reader = Data_Reader('./images_knee/', './labels_knee/', 1, 2)
while True:
    image, label = data_reader.getBatch()
    print(image.shape, label.shape)
    #for i in range(image.shape[0]):
    for j in range(image.shape[0]):
        temp = image[j,:,:,0].astype(np.uint8)
        temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
        lab = label[j,:,:,1]
        temp[lab > 0] = [0,0,255]

        lab = label[j,:,:,2]
        temp[lab > 0] = [0,255,255]
        """
        lab = label[i,:,:,6*j+3]
        temp[lab > 0] = [255,0,0]
        lab = label[i,:,:,6*j+4]
        temp[lab > 0] = [0,255,0]
        lab = label[i,:,:,6*j+5]
        temp[lab > 0] = [128,200,128]
        """
        cv2.imshow('res', temp)
        cv2.waitKey(0)

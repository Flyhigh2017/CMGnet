import tensorflow as tf
import numpy as np
import os
import sys
import cv2
from Data_Reader import Data_Reader
from segnet import Segnet
import os
from os import listdir
import time
#patient = '61040_20171020_CT_202_317_fnfL3'
data_dir="./AVN/test/test_images"# Images and labels for training

logs_dir = './logs/'
save_dir = './run_result/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
def softmax(x):
    sum_ex = np.sum(np.exp(x), axis=2)
    repeated = np.zeros(x.shape)
    for i in range(x.shape[2]):
        repeated[:,:,i] = sum_ex
    return np.exp(x)/repeated

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
def dice_loss(y_true, y_predicted):
    epsilon = 1e-6
    num_sum = 2.0 * tf.reduce_sum(y_true * y_predicted) + epsilon
    den_sum = tf.reduce_sum(y_true) + tf.reduce_sum(y_predicted) + epsilon
    return 1 - num_sum / (den_sum)

def loss_layer(predict, labels, class_weights=np.array([[1.0, 1.2]]).T.astype(np.float32),
               scope='loss_layer'):
    predict = tf.reshape(predict, (-1, 2))
    valid_predict = tf.nn.sigmoid(predict)
    valid_labels = tf.reshape(labels, (-1, 2))
    dice_L = dice_loss(valid_labels[:, 1], valid_predict[:, 1])
    loss = dice_L
    return loss
def main():
    t0 = time.time()
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(logs_dir + 'test.pb', "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    with tf.Session(config=config, graph=graph) as sess:
        x = sess.graph.get_tensor_by_name("import/images:0")
        model1 = sess.graph.get_tensor_by_name("import/output1:0")
        flag = sess.graph.get_tensor_by_name("import/train:0")
        lab = sess.graph.get_tensor_by_name("import/label:0")
        l = sess.graph.get_tensor_by_name("import/total_loss:0")
        '''
        loss = loss_layer(model1, lab)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
            train_op = optimizer.minimize(loss)
        '''
        print("load network: ", time.time() - t0)
        print (len(listdir(data_dir)))
        t0 = time.time()
        lst = listdir(data_dir)

        for im in lst:
            img = cv2.imread(data_dir + '/' + im, 0)
            img = cv2.resize(img, (128, 256))
            temp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            image = np.zeros((1, 256, 128, 1)).astype(np.uint8)
            image[0,:,:,0] = img
            label = np.zeros((1, 256, 128, 1)).astype(np.uint8)
            logits, losses = sess.run([model1, l], {x: image, flag: True, lab: label})
            print(losses)
            pred = softmax(logits[0,:,:,:])
            #print(pred.shape)
            channel1 = np.argmax(pred, axis=2)
            temp[channel1 == 1] = [0,0,255]
            cv2.imshow('res', temp.astype(np.uint8))
            cv2.waitKey(0)
            cv2.imwrite(save_dir + im + '.jpg', temp.astype(np.uint8))
        """
        for j in range(Batch_Size):

            img = Images[j,:,:,0].astype(np.uint8)
            temp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            pred = softmax(logits[j,:,:,:])
            #print(pred.shape)
            channel1 = np.argmax(pred, axis=2)
            temp[channel1 == 1] = [0,0,255]
            #temp[channel1 == 2] = [0,255,0]
            #temp[channel1 == 3] = [255,0,0]
            #temp[channel1 == 4] = [0,255,0]
            #temp[channel1 == 5] = [128,200,128]

            #im_color[channel1 > 0.5] = [0, 0, 255]
            cv2.imwrite(save_dir + '_' + str(itr) + '_' + str(j) + '.jpg', temp.astype(np.uint8))
            # im_color[channel1 > 0.5] = [0, 0, 255]
        """

if __name__ == '__main__':
    main()


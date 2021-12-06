import tensorflow as tf
import numpy as np
import os
import sys
from Data_ReaderFemur import Data_Reader
from segnet_atrous import Segnet
import os
from os import listdir
from tensorflow.python.framework import graph_util
import cv2
learning_rate=1e-4 #Learning rate for Adam Optimizer
Batch_Size=3 # Number of files per training iteration

MAX_ITERATION = int(89400) # Max  number of training iteration
logs_dir = './logs/'
save_dir = './save/'
save_iter = 100

if not os.path.exists(logs_dir):
    os.mkdir(logs_dir)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def softmax(x):
    sum_ex = np.sum(np.exp(x), axis=2)
    #print(sum_ex.shape)
    repeated = np.zeros(x.shape)
    for i in range(x.shape[2]):
        repeated[:,:,i] = sum_ex
    return np.exp(x)/repeated

def main(argv=None):
    tf.reset_default_graph()
    data_reader = Data_Reader('./AVN/IMG/', './AVN/LAB/', Batch_Size, 2)
    segnet = Segnet(True, 2)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(segnet.total_loss)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    print("Setting up Saver...")

    for itr in range(MAX_ITERATION):
        Images, GTLabels = data_reader.getBatch()

        feed_dict = {segnet.images: Images, segnet.label: GTLabels, segnet.train: False}
        total_loss, logits = sess.run([segnet.total_loss, segnet.logits],
                                                   feed_dict=feed_dict)  # Train one cycle

        if total_loss > 0.03:
            print("#################################TRAIN COARSE SEGMENTATION NET######################################")
            feed_dict = {segnet.images: Images, segnet.label: GTLabels, segnet.train: True}
            _, total_loss, logits = sess.run([train_op, segnet.total_loss, segnet.logits], feed_dict=feed_dict)
            print("train loss:", total_loss, "itr:", itr)
            for j in range(Batch_Size):
                img = Images[j, :, :, 0].astype(np.uint8)
                temp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                pred = softmax(logits[j, :, :, :])
                # print(pred.shape)
                channel1 = np.argmax(pred, axis=2)
                temp[channel1 == 1] = [0, 0, 255]
                # temp[channel1 == 2] = [0,255,0]
                # temp[channel1 == 3] = [255,0,0]
                # temp[channel1 == 4] = [0,255,0]
                # temp[channel1 == 5] = [128,200,128]

                # im_color[channel1 > 0.5] = [0, 0, 255]
                cv2.imwrite(save_dir + str(itr) + '_' + str(j) + '.jpg', temp.astype(np.uint8))

        if itr % save_iter == 0 and itr > 0:
            #constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["hour_glass1/output1"])
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output1", "total_loss"])
            with tf.gfile.FastGFile(logs_dir + 'test.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())

main()#Run script
print("Finished")

# ADAPTED BY Rafael Rego Drumond and Lukas Brinkmeyer
# THIS IMPLEMENTATION USES THE CODE FROM: https://github.com/dragen1860/MAML-TensorFlow

import numpy as np
import tensorflow as tf
from archs.maml2 import MAML
def getBin(l=10):
    x_ = 2
    n = 1
    while x_ < l:
        x_ = x_* 2
        n += 1
    
    numbers = []
    for i in range(l):
        num = []
        for j in list('{0:0b}'.format(i+1).zfill(n)):
            num.append(int(j))
        numbers.append(num)
    return numbers
class Model(MAML):
    def __init__(self,train_lr,meta_lr,image_shape,isMIN, label_size=2):
        super().__init__(train_lr,meta_lr,image_shape,isMIN, label_size)
        self.finals  = 64
        if isMIN:
            self.finals  = 800
    def getBin(self, l=10):
        x_ = 2
        n = 1
        while x_ < l:
            x_ = x_* 2
            n += 1

        numbers = []
        for i in range(l):
            num = []
            for j in list('{0:0b}'.format(i+1).zfill(n)):
                num.append(int(j))
            numbers.append(num)
        return numbers

    def dense_weights(self):
        weights     = {}
        cells       = {}
        initializer = tf.contrib.layers.xavier_initializer()
        divider = 1
        inic    = 1
        filters = 64
        self.finals  = 64
        if self.isMIN:
            print("\n\n\n\n\n\n\n\n\nIS MIN\n\n\n\n\n\n\n\n\n\n\n")
            divider = 2
            inic    = 3
            self.finals  = 800 
            filters = 32
        with tf.variable_scope('MASTER', reuse= tf.AUTO_REUSE):
            cells['d_1']  = tf.get_variable('MASTER_d_1w', [self.finals,1], initializer = initializer)
            cells['b_1']  = tf.get_variable('MASTER_d_1b', [1],   initializer=tf.initializers.constant)
        with tf.variable_scope('MAML', reuse= tf.AUTO_REUSE):
            weights['c_1'] = tf.get_variable('c_1', shape=(3,3, inic,filters), initializer=initializer)
            weights['c_2']  = tf.get_variable('c_2',  shape=(3,3,filters,filters), initializer=initializer)
            weights['c_3']  = tf.get_variable('c_3',  shape=(3,3,filters,filters), initializer=initializer)
            weights['c_4']  = tf.get_variable('c_4',  shape=(3,3,filters,filters), initializer=initializer)
            weights['cb_1'] = tf.get_variable('cb_1', shape=(filters), initializer=tf.initializers.constant)
            weights['cb_2'] = tf.get_variable('cb_2', shape=(filters), initializer=tf.initializers.constant)
            weights['cb_3'] = tf.get_variable('cb_3', shape=(filters), initializer=tf.initializers.constant)
            weights['cb_4'] = tf.get_variable('cb_4', shape=(filters), initializer=tf.initializers.constant)
            for i in range (self.max_labels):
                weights['d_1w'+str(i)]  = tf.get_variable('d_1w'+str(i), [self.finals,1], initializer = initializer)
                weights['b_1w'+str(i)]  = tf.get_variable('d_1b'+str(i), [1],   initializer=tf.initializers.constant)
    

        return weights, cells

    def forward(self,x,weights, training):
        # with tf.variable_scope('MAML', reuse= tf.AUTO_REUSE):
        conv1 = self.conv_layer(x,    weights["c_1"],weights["cb_1"],"conv1")
        conv1 = tf.layers.batch_normalization(conv1, name="bn1", reuse=tf.AUTO_REUSE)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.layers.MaxPooling2D(2,2)(conv1)
        
        conv2 = self.conv_layer(conv1,weights["c_2"],weights["cb_2"],"conv2")
        conv2 = tf.layers.batch_normalization(conv2, name="bn2", reuse=tf.AUTO_REUSE)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.layers.MaxPooling2D(2,2)(conv2)
        
        conv3 = self.conv_layer(conv2,weights["c_3"],weights["cb_3"],"conv3")
        conv3 = tf.layers.batch_normalization(conv3, name="bn3", reuse=tf.AUTO_REUSE)
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.layers.MaxPooling2D(2,2)(conv3)
        
        conv4 = self.conv_layer(conv3,weights["c_4"],weights["cb_4"],"conv4")
        conv4 = tf.layers.batch_normalization(conv4, name="bn4", reuse=tf.AUTO_REUSE)
        conv4 = tf.nn.relu(conv4)
        conv4 = tf.layers.MaxPooling2D(2,2)(conv4)

        bn = tf.layers.Flatten()(conv4)

        agg = [self.fc_layer(bn,"dense"+str(i),weights["d_1w"+str(i)],weights["b_1w"+str(i)]) for i in range(self.max_labels)]
        fc1 = tf.concat(agg, axis=-1)[:,:self.label_n[0]]

        return fc1 
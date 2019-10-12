# ADAPTED BY Rafael Rego Drumond and Lukas Brinkmeyer
# THIS IMPLEMENTATION USES THE CODE FROM: https://github.com/dragen1860/MAML-TensorFlow

import os
import numpy as np
import tensorflow as tf
from archs.maml import MAML
class Model(MAML):
    def __init__(self,train_lr,meta_lr,image_shape,isMIN, label_size=2):
        super().__init__(train_lr,meta_lr,image_shape,isMIN,label_size)

    def dense_weights(self):
        weights     = {}
        cells       = {}
        initializer = tf.contrib.layers.xavier_initializer()
        print("Creating/loading Weights")
        divider = 1
        inic    = 1
        filters = 64
        finals  = 64
        if self.isMIN:
            divider = 2
            inic    = 3
            finals  = 800 
            filters = 32
        with tf.variable_scope('MAML', reuse= tf.AUTO_REUSE):
            weights['c_1'] = tf.get_variable('c_1', shape=(3,3, inic,filters), initializer=initializer)
            weights['c_2']  = tf.get_variable('c_2',  shape=(3,3,filters,filters), initializer=initializer)
            weights['c_3']  = tf.get_variable('c_3',  shape=(3,3,filters,filters), initializer=initializer)
            weights['c_4']  = tf.get_variable('c_4',  shape=(3,3,filters,filters), initializer=initializer)
            weights['cb_1'] = tf.get_variable('cb_1', shape=(filters), initializer=tf.initializers.constant)
            weights['cb_2'] = tf.get_variable('cb_2', shape=(filters), initializer=tf.initializers.constant)
            weights['cb_3'] = tf.get_variable('cb_3', shape=(filters), initializer=tf.initializers.constant)
            weights['cb_4'] = tf.get_variable('cb_4', shape=(filters), initializer=tf.initializers.constant)
            weights['d_1']  = tf.get_variable('d_1w', [finals,self.label_size], initializer = initializer)
            weights['b_1']  = tf.get_variable('d_1b', [self.label_size],   initializer=tf.initializers.constant)
           
            """weights['mean']     = tf.get_variable('mean',    [64],   initializer=tf.zeros_initializer())
            weights['variance'] = tf.get_variable('variance',[64],   initializer=tf.ones_initializer() )
            weights['offset']   = tf.get_variable('offset',  [64],   initializer=tf.zeros_initializer())
            weights['scale']    = tf.get_variable('scale',   [64],   initializer=tf.ones_initializer() )
            
            weights['mean1']     = tf.get_variable('mean',    [64],   initializer=tf.zeros_initializer())
            weights['variance1'] = tf.get_variable('variance',[64],   initializer=tf.ones_initializer() )
            weights['offset1']   = tf.get_variable('offset',  [64],   initializer=tf.zeros_initializer())
            weights['scale1']    = tf.get_variable('scale',   [64],   initializer=tf.ones_initializer() )
            
            weights['mean2']     = tf.get_variable('mean',    [64],   initializer=tf.zeros_initializer())
            weights['variance2'] = tf.get_variable('variance',[64],   initializer=tf.ones_initializer() )
            weights['offset2']   = tf.get_variable('offset',  [64],   initializer=tf.zeros_initializer())
            weights['scale2']    = tf.get_variable('scale',   [64],   initializer=tf.ones_initializer() )
            
            weights['mean3']     = tf.get_variable('mean',    [64],   initializer=tf.zeros_initializer())
            weights['variance3'] = tf.get_variable('variance',[64],   initializer=tf.ones_initializer() )
            weights['offset3']   = tf.get_variable('offset',  [64],   initializer=tf.zeros_initializer())
            weights['scale3']    = tf.get_variable('scale',   [64],   initializer=tf.ones_initializer() )"""
            print("Done Creating/loading Weights")
        return weights, cells
    
    def forward(self,x,weights, training):
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
        # print(conv4)
#         bn = tf.squeeze(conv4,axis=(1,2))
        bn = tf.layers.Flatten()(conv4)
        # tf.reshape(bn, [3244,234])

        fc1 = self.fc_layer(bn,"dense1",weights["d_1"],weights["b_1"])
#         bn = tf.reshape(bn,[-1,])
        return  fc1
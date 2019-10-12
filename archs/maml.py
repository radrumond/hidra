# ADAPTED BY Rafael Rego Drumond and Lukas Brinkmeyer
# THIS IMPLEMENTATION USES THE CODE FROM: https://github.com/dragen1860/MAML-TensorFlow

import os
import numpy as np
import tensorflow as tf

class MAML:
    def __init__(self,train_lr,meta_lr,image_shape, isMIN, label_size=2):
        self.train_lr = train_lr
        self.meta_lr  = meta_lr
        self.image_shape = image_shape
        self.isMIN = isMIN
        self.saver = None
        self.label_size = label_size
        self.finals  = 64
        self.maml_n = 1
        if isMIN:
            self.finals  = 800
    def build(self, K, meta_batchsz, mode='train'):
        
        # Meta batch of tasks 
        self.train_xb = tf.placeholder(tf.float32, [None,None,None,None,self.image_shape[-1]])
        self.train_yb = tf.placeholder(tf.float32, [None,None,None])
        self.val_xb   = tf.placeholder(tf.float32, [None,None,None,None,self.image_shape[-1]])
        self.val_yb   = tf.placeholder(tf.float32, [None,None,None])
        self.label_n  = tf.placeholder(tf.int32    , 1, name="num_labs")    
        #Initialize weights
        self.weights, self.cells = self.dense_weights()
        training = True if mode is 'train' else False
        
        # Handle one task update
        def meta_task(inputs):
            train_x, train_y, val_x, val_y = inputs
            val_preds, val_losses = [], []

            train_pred = self.forward(train_x, self.weights, training)
            train_loss = tf.losses.softmax_cross_entropy(train_y,train_pred)
            
            grads = tf.gradients(train_loss, list(self.weights.values()))
            gvs = dict(zip(self.weights.keys(), grads))
            
            a=[self.weights[key] - self.train_lr * gvs[key] for key in self.weights.keys()]
#             for key in self.weights.keys():
#                 print(key, gvs[key])
            fast_weights = dict(zip(self.weights.keys(),a))

            # Validation after each update
            val_pred = self.forward(val_x, fast_weights, training)
            val_loss = tf.losses.softmax_cross_entropy(val_y,val_pred)
            # record T0 pred and loss for meta-test
            val_preds.append(val_pred)
            val_losses.append(val_loss)
            
            # continue to build T1-TK steps graph
            for _ in range(1, K):
    
                # Update weights on train data of task t
                loss =  tf.losses.softmax_cross_entropy(train_y,self.forward(train_x, fast_weights, training))
                grads = tf.gradients(loss, list(fast_weights.values()))
                gvs = dict(zip(fast_weights.keys(), grads))
                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.train_lr * gvs[key] for key in fast_weights.keys()]))
                
                # Evaluate validation data of task t
                val_pred = self.forward(val_x, fast_weights, training)
                val_loss = tf.losses.softmax_cross_entropy(val_y,val_pred)
                val_preds.append(val_pred)
                val_losses.append(val_loss)

            result = [train_pred, train_loss, val_preds, val_losses]

            return result
        
        out_dtype = [tf.float32, tf.float32,[tf.float32] * K, [tf.float32] * K]
        result = tf.map_fn(meta_task, elems=(self.train_xb, self.train_yb, self.val_xb, self.val_yb),
                           dtype=out_dtype, parallel_iterations=meta_batchsz, name='map_fn')
        train_pred_tasks, train_loss_tasks, val_preds_tasks, val_losses_tasks = result

        if mode is 'train':
            self.train_loss = train_loss = tf.reduce_sum(train_loss_tasks) / meta_batchsz
            self.val_losses = val_losses = [tf.reduce_sum(val_losses_tasks[j]) / meta_batchsz for j in range(K)]
            self.val_predictions = val_preds_tasks
            
            optimizer = tf.train.AdamOptimizer(self.meta_lr, name='meta_optim')
            gvs = optimizer.compute_gradients(self.val_losses[-1])
            gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
            self.meta_op = optimizer.apply_gradients(gvs)

        else: 
            self.test_train_loss = train_loss = tf.reduce_sum(train_loss_tasks) / meta_batchsz
            self.test_val_losses = val_losses = [tf.reduce_sum(val_losses_tasks[j]) / meta_batchsz for j in range(K)]
            self.val_predictions = val_preds_tasks

        self.saving_weights = tf.trainable_variables()
    def conv_layer(self, x, W, b, name, strides=1):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
            x = tf.nn.bias_add(x, b)
        return x

    def fc_layer(self,x, name, weights=None, biases=None):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
                fc = tf.matmul(x, weights)
                fc = tf.nn.bias_add(fc, biases)
                return fc

    def loadWeights(self, sess, name, step=0, modeldir='./model_checkpoint/', model_name='model.ckpt'):
        if self.saver == None:
            z = self.saving_weights
            #print("KEYS:", z.keys())
            self.saver = tf.train.Saver(var_list=z, max_to_keep=12)
        saver = self.saver
        checkpoint_path = modeldir + f"{name}/"+model_name +"-" + step
        if os.path.isfile(checkpoint_path+".marker"):
            saver.restore(sess, checkpoint_path)
            print('The checkpoint has been loaded.')
        else:
            print(checkpoint_path+".marker not found. Starting from scratch.")
            
    def saveWeights(self, sess, name, step=0, modeldir='./model_checkpoint/', model_name='model.ckpt'):
        if self.saver == None:
            z = self.saving_weights
            self.saver = tf.train.Saver(var_list=z, max_to_keep=12)
        saver = self.saver
        checkpoint_path = modeldir + f"{name}/"+model_name
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
        saver.save(sess, checkpoint_path, global_step=step)
        print('The checkpoint has been created.')
        open(checkpoint_path+"-"+str(int(step))+".marker", 'a').close()
    

    def dense_weights(self):
        return
    def forward(self,x,weights, training):
        return
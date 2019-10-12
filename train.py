import numpy as np
import tensorflow as tf
from data_gen.omni_gen import unison_shuffled_copies,OmniChar_Gen, MiniImgNet_Gen
import time

def train(   m, mt,     # m is the model foir training, mt is the model for testing
          data_sampler, # Creates the data generator for training and testing
          min_classes,  # minimum amount of classes
          max_classes,  # maximum    ||  ||   ||
          train_shots,  # number of samples per class (train)
          test_shots,   # number of samples per class (test)
          meta_batch,   # Number of tasks
          meta_iters,   # Number of iterations
          test_iters,   # Iterations in Test
          train_step,
          name):        # Experiment name for experiments
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # bnorms = [v for v in tf.global_variables() if "bn" in v.name]
    #---------Performance Tracking lists---------------------------------------
    losses  = []
    temp_yp = []
    temp_ypn= []
    nls     = []
    aps     = []
    buffer  = []
    lossesB = []
    #--------------------------------------------------------------------------

    #---------Load train and test data-sets------------------------------------
    train_gen = data_sampler.sample_Task(meta_batch,min_classes,max_classes+1,train_shots,test_shots,"train")
    if mt is not None:
        test_gen  = data_sampler.sample_Task(meta_batch,min_classes,max_classes+1,train_shots,test_shots,"test" )
    m.loadWeights(sess, name, step=str(int(train_step)), model_name=name+".ckpt")
    #--------------------------------------------------------------------------

    #TRAIN LOOP
    print("Starting meta training:")
    start = time.time()
    for i in range(meta_iters):

        xb1,yb1,xb2,yb2 = next(train_gen)
        num_l = [len(np.unique(np.argmax(yb1,axis=-1)))]

        if m.maml_n == 2: # in case it uses hydra master node, it should re-assign the output nodes from the master
            sess.run(m.init_assign, feed_dict={m.label_n:[5]})
        l,_,vals,ps=sess.run([m.train_loss,m.meta_op,m.val_losses,m.val_predictions],feed_dict={m.train_xb: xb1,
                                                                                                m.train_yb: yb1,
                                                                                                m.val_xb:xb2,
                                                                                                m.val_yb:yb2,
                                                                                                m.label_n:num_l})
        if m.maml_n == 2: # in case it uses hydra master node, it should update the master
            sess.run(m.final_assign,feed_dict={m.label_n:num_l})

        losses.append(vals)
        lossesB.append(vals)
        buffer.append(l)

        #Calculate accuaracies
        aux = []
        tmp_pred = np.argmax(np.reshape(ps[-1],[-1,num_l[0]]),axis=-1)
        tmp_true = np.argmax(np.reshape(yb2,[-1,num_l[0]]),axis=-1)
        for ccci in range(num_l[0]):
            tmp_idx = np.where(tmp_true==ccci)[0]
            #print(tmp_idx)
            aux.append(np.mean(tmp_pred[tmp_idx]==tmp_true[tmp_idx]))
        temp_yp.append(np.mean(tmp_pred==tmp_true))
        temp_ypn.append(aux)

        #EVALUATE and PRINT
        if i%100==0:
            testString = ""
            #If we give a test model, it will test using the weights from train
            if mt is not None and i%1000==0:
                lossestest  = []
                buffertest  = []
                lossesBtest = []
                temp_yptest = []
                for z in range(100):
                    if m.maml_n == 2:
                        sess.run(mt.init_assign, feed_dict={mt.label_n:[5]})
                    xb1,yb1,xb2,yb2 = next(test_gen)
                    num_l = [len(np.unique(np.argmax(yb1,axis=-1)))]
                    l,vals,ps=sess.run([mt.test_train_loss,mt.test_val_losses,mt.val_predictions],feed_dict={mt.train_xb: xb1,
                                                                                                    mt.train_yb: yb1,
                                                                                                    mt.val_xb:xb2,
                                                                                                    mt.val_yb:yb2,
                                                                                                    mt.label_n:num_l})
                    lossestest.append(vals)
                    lossesBtest.append(vals)
                    buffertest.append(l)
                    temp_yptest.append(np.mean(np.argmax(ps[-1],axis=-1)==np.argmax(yb2,axis=-1)))
                
                testString = f"\n        TEST: TLoss {np.mean(buffertest):.3f} VLoss {np.mean(lossesBtest,axis=0)[-1]:.3f}, ACCURACY {np.mean(temp_yptest):.4f}"
            print(f"Epoch {i}: TLoss {np.mean(buffer):.4f}, VLoss {np.mean(lossesB,axis=0)[-1]:.4f},",
                  f"Accuracy {np.mean(temp_yp):.4}", f", Per label acc: {[float('%.4f' % elem) for elem in aux]}", f"Finished in {time.time()-start}s",testString)

            buffer  = []
            lossesB = []
            temp_yp = []
            start = time.time()
                  # f"\n TRUE: {yb2}\n PRED: {ps}")
        if i%5000==0:
            print("Saving...")
            m.saveWeights(sess, name, i, model_name=name+".ckpt")

    m.saveWeights(sess, name, i, model_name=name+".ckpt")

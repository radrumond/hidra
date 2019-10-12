import numpy as np
import tensorflow as tf
from data_gen.omni_gen import unison_shuffled_copies,OmniChar_Gen, MiniImgNet_Gen

def test(m, data_sampler,
          eval_step,
          min_classes,
          max_classes,
          train_shots,
          test_shots,
          meta_batch,
          meta_iters,
          name):
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    losses=[]

    temp_yp = []
    aps = []
    buffer = []
    lossesB=[]

    train_gen = data_sampler.sample_Task(meta_batch,min_classes,max_classes+1,train_shots,test_shots,"test")
    print("TEST MODE")
    m.loadWeights(sess, name, step = str(int(eval_step)), model_name=name+".ckpt")
    for i in range(meta_iters):
        xb1,yb1,xb2,yb2 = next(train_gen)
        num_l = [len(np.unique(np.argmax(yb1,axis=-1)))]

        if m.maml_n == 2:
            sess.run(m.init_assign, feed_dict={m.label_n:[5]})
        l,vals,ps=sess.run([m.test_train_loss,m.test_val_losses,m.val_predictions],feed_dict={m.train_xb: xb1,
                                                                                                m.train_yb: yb1,
                                                                                                m.val_xb:xb2,
                                                                                                m.val_yb:yb2,
                                                                                                m.label_n:num_l})

        losses.append(vals)
        lossesB.append(vals)
        buffer.append(l)

        true_vals = np.argmax(yb2,axis=-1)
        all_accs = []
        for pred_epoch in range(len(ps)):
        	all_accs.append(np.mean(np.argmax(ps[pred_epoch],axis=-1)==true_vals))
        temp_yp.append(all_accs)


        # if i%1==0:
        if i%50==0:
            print(f"({i}/{meta_iters})")
            print(f"Final: TLoss {np.mean(buffer)}, VLoss {np.mean(lossesB,axis=0)}", f"Accuracy {np.mean(temp_yp,axis=0)}" )
    print(f"Final: TLoss {np.mean(buffer)}-{np.std(buffer)}, VLoss {np.mean(lossesB,axis=0)}-{np.std(lossesB,axis=0)}", f"Accuracy {np.mean(temp_yp,axis=0)}-{np.std(temp_yp,axis=0)}" )


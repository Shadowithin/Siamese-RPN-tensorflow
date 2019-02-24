import tensorflow as tf
from net.Siamese import SiameseRPN
from utils.image_reader_cuda import Image_reader
from module.loss_module import Loss_op
import os
import numpy as np
import cv2
from module.gen_ancor import Anchor
from module.debug import debug
from config import cfg
import time

class Train():
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        self.reader=Image_reader(cfg.root_dir)
        self.step_num=self.reader.img_num*cfg.epoch_num
        self.save_per_epoch=self.reader.img_num
        print(self.reader.img_num)
        self.loss_op=Loss_op()
        self.learning_rate=cfg.learning_rate
        self.decay_rate=cfg.decay_rate
        self.decay_step=int(self.save_per_epoch/4)
        self.model_dir=cfg.model_dir
        self.pre_trained_dir=cfg.pre_trained_dir
        self.anchor_op=Anchor(49,49)
        self.is_debug=True

    def train(self):

        template,_,detection,gt_box,_,_=self.reader.get_batch()
        net=SiameseRPN({'template':template,'detection':detection})

        pre_cls=net.layers['cls']
        pre_reg=net.layers['reg']

        cls_loss,reg_loss,label,target_box=self.loss_op.loss(gt_box[0],pre_cls,pre_reg)
        loss=cls_loss+reg_loss

        #+++++++++++++++++++++debug++++++++++++++++++++++++++++++
        with tf.name_scope('debug'):
            pre_cls_shape = tf.shape(pre_cls)
            debug_pre_cls=tf.reshape(pre_cls,(pre_cls_shape[1],pre_cls_shape[2],net.k,2))# 17*17*5*2
            debug_pre_cls=tf.expand_dims(tf.transpose(tf.nn.softmax(debug_pre_cls)[:,:,:,1], (2,0,1)),axis=-1)# score map,5*17*17*1
            debug_pre_reg=pre_reg
            debug_pre_score=tf.nn.softmax(tf.reshape(pre_cls,(-1,2)))
            debug_pre_box=tf.reshape(pre_reg,(-1,4))
        #+++++++++++++++++++++debug++++++++++++++++++++++++++++++

        with tf.name_scope('train_op'):
            global_step=tf.Variable(0,trainable=False)
            lr=tf.train.exponential_decay(0.001,global_step,self.decay_step,self.decay_rate,staircase=True,name='lr')
            train_op=tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step)

        with tf.name_scope('summary'):
            tf.summary.scalar('cls_loss', cls_loss)
            tf.summary.scalar('reg_loss', reg_loss)
            tf.summary.scalar('total_loss', loss)
            tf.summary.scalar('learning_rate', lr)
            tf.summary.histogram('pre_score', debug_pre_score)
            tf.summary.image('score_map', debug_pre_cls, max_outputs=5)
            merge_summary=tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=50)
        coord=tf.train.Coordinator()
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess=tf.InteractiveSession(config=config)
        threads=tf.train.start_queue_runners(coord=coord,sess=sess)
        summary_writer=tf.summary.FileWriter('logs', sess.graph)
        sess.run(tf.global_variables_initializer())



        if self.load(saver,sess,self.model_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        epoch=0
        t=time.time()
        step = 0
        while (step < self.step_num):
            #+++++++++++++++++++++debug++++++++++++++++++++++++++++++
            if self.is_debug:
                _,summaries,loss_,cls_loss_,reg_loss_,lr_,debug_pre_cls_,debug_pre_reg_,debug_pre_score_,debug_pre_box_,label_,target_box_,detection_p,template_p,detection_label_p,g_step=\
                sess.run([train_op,merge_summary,loss,cls_loss,reg_loss,lr,debug_pre_cls,debug_pre_reg,debug_pre_score,debug_pre_box,label,target_box,detection,template,gt_box,global_step])
                if step %500==0:
                    debug(detection_p[0],template_p[0],detection_label_p[0],debug_pre_cls_,debug_pre_reg_,debug_pre_score_,debug_pre_box_,label_,target_box_,step,self.anchor_op)
                    summary_writer.add_summary(summaries, global_step=g_step)
            #+++++++++++++++++++++debug++++++++++++++++++++++++++++++
            else:
                _,loss_,cls_loss_,reg_loss_,lr_,g_step=sess.run([train_op,loss,cls_loss,reg_loss,lr,global_step])
            if step == 0:
                step+=g_step
            if step %100==0:
                print('step={},loss={},cls_loss={},reg_loss={},lr={},time={}'.format(step,loss_,cls_loss_,reg_loss_,lr_,time.time()-t))
                t=time.time()
            if step %self.save_per_epoch==0 and step>0:
                epoch+=1
                self.save(saver,sess,self.model_dir,g_step)
            step+=1

        coord.request_stop()
        coord.join(threads)

    def load(self,saver,sess,ckpt_path):
        ckpt=tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name=os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess,os.path.join(ckpt_path,ckpt_name))
            print("Restored model parameters from {}".format(ckpt_name))
            return True
        else:
            return False
    def save(self,saver,sess,ckpt_path,global_step):
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        model_path=os.path.join(ckpt_path,'model')
        saver.save(sess,model_path,global_step)
        print('saved model')
if __name__=='__main__':
    t=Train()
    t.train()



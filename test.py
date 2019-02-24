import tensorflow as tf
from net.Siamese_forward import SiameseRPN
from utils.image_reader_forward import Image_reader
import os
import numpy as np
import cv2
from module.gen_ancor import Anchor
from config import cfg
import sys
class Test():
    def __init__(self):
        if len(sys.argv) > 1:
            self.img_path = sys.argv[1]
            self.reader = Image_reader(img_path=self.img_path, label_path=self.img_path+'/groundtruth.txt')
        else:
            self.img_path = cfg.img_path
            self.reader=Image_reader(img_path=cfg.img_path,label_path=cfg.label_path)
        self.model_dir=cfg.model_dir
        self.anchor_op=Anchor(49,49)
        self.anchors=self.anchor_op.anchors
        self.anchors=self.anchor_op.regu()
        self.anchors=self.anchor_op.corner_to_center(self.anchors)
        self.penalty_k=cfg.penalty_k
        self.window_influence=cfg.window_influence
        self.lr=cfg.lr
        self.vedio_dir = cfg.vedio_dir
        if len(sys.argv) > 1:
            self.vedio_name = sys.argv[1].split('/')[-1]+'_out.mp4'
        else:
            self.vedio_name=cfg.vedio_name
    def test(self):
        #===================input-output====================
        img_t=tf.placeholder(dtype=tf.float32,shape=[1,None,None,3])
        conv_c=tf.placeholder(dtype=tf.float32,shape=[4,4,256,10])
        conv_r=tf.placeholder(dtype=tf.float32,shape=[4,4,256,20])

        net=SiameseRPN({'img':img_t,'conv_c':conv_c,'conv_r':conv_r})

        pre_conv_c=net.layers['t_c_k']
        pre_conv_r=net.layers['t_r_k']

        pre_cls=net.layers['cls']
        pre_reg=net.layers['reg']
        pre_cls=tf.nn.softmax(tf.reshape(pre_cls,(-1,2)))
        pre_reg=tf.reshape(pre_reg,(-1,4))
        conv_r_=np.zeros((4,4,256,20))
        conv_c_=np.zeros((4,4,256,10))
        box_ori=None
        #===================input-output====================

        #======================hanning======================
        # w = np.outer(np.hanning(17), np.hanning(17))
        # w=np.stack([w,w,w,w,w],-1)
        # self.window=w.reshape((-1))
        #======================hanning======================

        #================start-tensorflow===================
        loader=tf.train.Saver()
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess=tf.InteractiveSession(config=config)
        sess.run(tf.global_variables_initializer())
        if self.load(loader,sess,self.model_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        #================start-tensorflow===================

        frames=[]
        for step in range(self.reader.img_num):
            img,box,img_p,box_p,offset,ratio=self.reader.get_data(frame_n=step,pre_box=box_ori)
            #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            img_h, img_w, _ = img.shape
            # videoWriter_box = cv2.VideoWriter(
            #     os.path.join(self.vedio_dir, self.vedio_name.split('.')[0] + '_box.' + self.vedio_name.split('.')[1]),
            #     fourcc, 30, (512, 512))
            # videoWriter_box = cv2.VideoWriter("test.mp4", fourcc, 30, (512,512))

            if step==0:
                img_p_show = cv2.cvtColor((img_p*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
                cv2.imshow("template", img_p_show)
                if not os.path.exists(self.img_path+'/results'):
                    os.makedirs(self.img_path+'/results')
                cv2.imwrite(self.img_path+'/results/template.jpg', img_p_show)
            img_p=np.expand_dims(img_p,axis=0)
            feed_dict={img_t:img_p,conv_c:conv_c_,conv_r:conv_r_, net.frame: step}
            if step==0:
                #init
                conv_c_,conv_r_=sess.run([pre_conv_c,pre_conv_r],feed_dict=feed_dict)
                box_ori = box
                #pre_box=box_ori#[x,y,w,h]===x,y is left-top corner
            else:
                frames.append(img[:,:,::-1])
                pre_cls_,pre_reg_=sess.run([pre_cls,pre_reg],feed_dict=feed_dict)
                bboxes=self.nms(pre_cls_,pre_reg_)
                #pre_box=self.recover(bbox,offset,ratio,pre_box)#[x1,y1,x2,y2]
                img_p_show = cv2.cvtColor((img_p[0]*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                for bbox in bboxes:
                    color = np.random.random((3,)) * 0.6 + 0.4
                    color = color * 255
                    color = color.astype(np.int32).tolist()
                    cv2.rectangle(img_p_show, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
                cv2.rectangle(img_p_show, (int(bboxes[0][0]), int(bboxes[0][1])), (int(bboxes[0][2]), int(bboxes[0][3])), (0, 0, 0), 2)


                # #+++++++++++++++++++++gt_box++++++++++++++++++++++++++++++
                # box_ori[2]=box_ori[0]+box_ori[2]
                # box_ori[3]=box_ori[1]+box_ori[3]
                # img=cv2.rectangle(img,(int(box_ori[0]),int(box_ori[1])),(int(box_ori[2]),int(box_ori[3])),(0,0,0),1)
                # #+++++++++++++++++++++gt_box++++++++++++++++++++++++++++++

                #+++++++++++++++++++++gt_box++++++++++++++++++++++++++++++
                # box_ori[2]=box_ori[0]+box_ori[2]
                # box_ori[3]=box_ori[1]+box_ori[3]
                # img=cv2.rectangle(img,(int(box_ori[0]),int(box_ori[1])),(int(box_ori[2]),int(box_ori[3])),(0,0,0),1)
                #+++++++++++++++++++++gt_box++++++++++++++++++++++++++++++

                #videoWriter_box.write(img_p_show)
                cv2.imshow('img',img_p_show)
                cv2.imwrite(self.img_path+"/results/%04d.jpg"%step, img_p_show)
                cv2.waitKey(10)

                #
                # pre_box[2]=pre_box[2]-pre_box[0]
                # pre_box[3]=pre_box[3]-pre_box[1]
        print('video are being synthesized. please wait for one minute..............')
        #videoWriter.release()
        #videoWriter_box.release()
        print('vedio is saved in '+self.vedio_dir)

    def nms(self,scores,delta):
        score=scores[:,1]
        index_score = np.argsort(scores[:, 1])[::-1][0:5]
        boxes = delta[index_score]
        anchors = self.anchors[index_score]

        bboxes=np.zeros_like(boxes)
        bboxes[:,0]=boxes[:,0]*anchors[:,2]+anchors[:,0]
        bboxes[:,1]=boxes[:,1]*anchors[:,3]+anchors[:,1]
        bboxes[:,2]=np.exp(boxes[:,2])*anchors[:,2]
        bboxes[:,3]=np.exp(boxes[:,3])*anchors[:,3]#[x,y,w,h]
        # def change(r):
        #     return np.maximum(r, 1./r)
        # def sz(w, h):
        #     pad = (w + h) * 0.5
        #     sz2 = (w + pad) * (h + pad)
        #     return np.sqrt(sz2)
        # def sz_wh(wh):
        #     pad = (wh[0] + wh[1]) * 0.5
        #     sz2 = (wh[0] + pad) * (wh[1] + pad)
        #     return np.sqrt(sz2)

        # size penalty
        # s_c = change(sz(bboxes[:,2], bboxes[:,3]) / (sz_wh(target_sz)))  # scale penalty
        # r_c = change((target_sz[0] / target_sz[1]) / (bboxes[:,2] / bboxes[:,3]))  # ratio penalty
        #
        # penalty = np.exp(-(r_c * s_c - 1.) * self.penalty_k)
        # pscore = penalty * score

        # window float
        # pscore = pscore * (1 - self.window_influence) + self.window * self.window_influence
        #best_pscore_id = np.argmax(score)

        #self.lr = penalty[best_pscore_id] * score[best_pscore_id] * self.lr
        #bbox=bboxes[best_pscore_id].reshape((1,4))#[x,y,w,h]
        bboxex = self.anchor_op.center_to_corner(bboxes)
        #+++++++++++++++++++++debug++++++++++++++++++++++++++++++
        # b=self.anchor_op.center_to_corner(bbox)
        # cv2.rectangle(img,(int(b[0][0]),int(b[0][1])),(int(b[0][2]),int(b[0][3])),(255,0,0),1)
        # cv2.imshow('resize',img)
        # cv2.waitKey(0)
        #+++++++++++++++++++++debug++++++++++++++++++++++++++++++

        return bboxex

    def recover(self,box,offset,ratio,pre_box):
        #label=[c_x,c_y,w,h]
        box[2]=box[2]*ratio
        box[3]=box[3]*ratio
        box[0]=box[0]*ratio+offset[0]
        box[1]=box[1]*ratio+offset[1]

        # box[2] = pre_box[2] * (1 - self.lr) + box[2] * self.lr
        # box[3] = pre_box[3] * (1 - self.lr) + box[3] * self.lr

        #centor to coner
        box[0]=int(box[0]-(box[2]-1)/2)
        box[1]=int(box[1]-(box[3]-1)/2)
        box[2]=round(box[0]+(box[2]))
        box[3]=round(box[1]+(box[3]))

        return box#[x1,y1,x2,y2]
    def load(self,saver,sess,ckpt_path):
        ckpt=tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name=os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess,os.path.join(ckpt_path,ckpt_name))
            print("Restored model parameters from {}".format(ckpt_name))
            return True
        else:
            return False

if __name__=='__main__':
    t=Test()
    t.test()


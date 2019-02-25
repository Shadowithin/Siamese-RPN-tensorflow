import numpy as np
import cv2
import os
CHANNEL_MEAN = [124, 117, 104]
class Image_reader():
    """docstring for Image_reader"""
    def __init__(self,img_path=None,label_path=None,mode='pic'):
        if mode=='pic':
            self.img_path=img_path
            self.label_path=label_path
            self.imgs=[]
            self.boxes=[]
            self.img_num=0
            #===========label===============
            with open(self.label_path) as f:
                line=f.readline().strip('\n')
                while (line):
                    box=line.split(',')
                    self.boxes.append([int(float(box[0])),int(float(box[1])),int(float(box[2])),int(float(box[3]))])
                    line=f.readline().strip('\n')
            #===========label===============

            #===========img_list============
            self.imgs=[x for x in os.listdir(self.img_path) if ('jpg' in x or 'png' in x or 'bmp' in x)]
            self.imgs.sort()
            self.img_num=len(self.imgs)
            #===========img_list============
        elif mode=='vedio':
            print('test vedio')
        else:
            print('error')

    def get_data(self,frame_n=0):
        img=cv2.imread(os.path.join(self.img_path,self.imgs[frame_n]))
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)# actually bgr 2 rgb
        #img_submean = np.subtract(img, CHANNEL_MEAN)
        try:
            box_ori=self.boxes[frame_n]#[x,y,w,h]===x,y is left-top corner
        except:
            box_ori = self.boxes[-1]
        #when current frame is the first frame, return the target img, else return detection img
        if frame_n==0:
            img_p,box_p,offset,ratio=self.crop_resize(img,box_ori,1)
        else:
            img_p,box_p,offset,ratio=self.crop_resize(img,box_ori,4)
        return img,box_ori,img_p,box_p,offset,ratio
    def get_vedio_data(self,img,box_ori=None,frame_n=0,pre_box=None,note=None):
        #[x,y,w,h]===x,y is left-top corner
        if frame_n==0:
            img_p,box_p,offset,ratio=self.crop_resize(img,box_ori,1,note=note)
        else:
            img_p,box_p,offset,ratio=self.crop_resize(img,pre_box,2,note=note)
        return img,box_ori,img_p,box_p,offset,ratio
    def recover(self,box,offset,ratio):
        #label=[c_x,c_y,w,h]
        box[2]=box[2]*ratio
        box[3]=box[3]*ratio
        box[0]=box[0]*ratio+offset[0]
        box[1]=box[1]*ratio+offset[1]

        box[0]=int(box[0]-(box[2]-1)/2)
        box[1]=int(box[1]-(box[3]-1)/2)
        box[2]=int(box[0]+(box[2]))
        box[3]=int(box[1]+(box[3]))

        return box

    def crop_resize(self,img,label,rate=1,random_patch=False,note=None):
        #label=[x,y,w,h]===x,y is left-top corner
        #print(label)
        x,y,w,h=label
        heigh,width=img.shape[:2]
        img=img.astype(np.float32)
        mean_axis=np.mean(img,axis=(0,1)).astype(np.float32)
        #===========================rectify==========================
        if not note is None and len(note)>0:
            note=np.array(note)
            c_x,c_y,c_w,c_h,score=note[:,0],note[:,1],note[:,2],note[:,2],note[:,4]
            if c_x[-1]<0 or c_x[-1]>width or c_y[-1]<0 or c_y[-1]>heigh:# or score[-1]<0.6:
                index=np.where(score>0.9)[0]
                print('rectify')
                if len(index)<5 and len(index)>0:
                    v_x=c_x[index]
                    v_y=c_y[index]
                    v_w=c_w[index]
                    v_h=c_h[index]
                    new_x=np.sum(v_x*([1/len(index)]*len(index)))
                    new_y=np.sum(v_y*([1/len(index)]*len(index)))
                    new_w=v_w[-1]
                    new_h=v_h[-1]
                elif len(index)>0:
                    index=index[-5:]
                    v_x=c_x[index]
                    v_y=c_y[index]
                    v_w=c_w[index]
                    v_h=c_h[index]
                    new_x=np.sum(v_x*[0.05,0.05,0.1,0.2,0.6])
                    new_y=np.sum(v_y*[0.05,0.05,0.1,0.2,0.6])
                    new_w=v_w[-1]
                    new_h=v_h[-1]
                else:
                    print('box is error')
                    new_x=x+w/2
                    new_y=y+h/2
                    new_w=w
                    new_h=h
                x=new_x-new_w/2
                y=new_y-new_h/2
                w=new_w
                h=new_h
        #===========================rectify==========================
        # p=(w+h)/2
        # s=(w+p)*(h+p)
        side=round(max(w,h)*rate)

        x1=int(x-int((side-w)/2))
        y1=int(y-int((side-h)/2))
        x2=int(x1+side)
        y2=int(y1+side)

        offset=np.zeros(2)
        offset[0]=x1-0
        offset[1]=y1-0

        if x1<0:
            img_offset=np.zeros((img.shape[0],-x1,3))+mean_axis
            img=np.hstack([img_offset,img])
            x-=x1
        if y1<0:
            img_offset=np.zeros((-y1,img.shape[1],3))+mean_axis
            img=np.vstack([img_offset,img])
            y-=y1
        if x2>width:
            img_offset=np.zeros((img.shape[0],x2-width+1,3))+mean_axis
            img=np.hstack([img,img_offset])
        if y2>heigh:
            img_offset=np.zeros((y2-heigh+1,img.shape[1],3))+mean_axis
            img=np.vstack([img,img_offset])

        x1=int(x-int((side-w)/2))
        y1=int(y-int((side-h)/2))
        x2=int(x1+side)
        y2=int(y1+side)
        crop_img = img[y1:y2, x1:x2, :]

        assert crop_img.shape[0]==side
        assert crop_img.shape[1]==side
        if rate==1:
            resize_img=cv2.resize(crop_img,(127,127))/255.
            ratio=side/127
            label=np.array([63,63,w/ratio,h/ratio]).astype(np.int32)
            label=label.astype(np.float32)
        if rate==4:
            if random_patch:
                shift_max_x = min(x-x1, img.shape[1]-x2-1)
                shift_min_x = -min(x1, x2-x-w)
                shift_max_y = min(y-y1, img.shape[0]-y2-1)
                shift_min_y = -min(y1, y2-y-h)

                random_x = np.random.uniform(shift_min_x, shift_max_x, size=[])
                random_y = np.random.uniform(shift_min_y, shift_max_y, size=[])

                x1 = int(x1 + random_x)
                x2 = int(x2 + random_x)
                y1 = int(y1 + random_y)
                y2 = int(y2 + random_y)

                offset = [x1, y1]
                crop_img = img[y1:y2, x1:x2, :]
                resize_img=cv2.resize(crop_img,(511,511))/255.
                ratio=side/255
                label=np.array([255-random_x/ratio,255-random_y/ratio,w/ratio,h/ratio]).astype(np.int32)
                label=label.astype(np.float32)
            else:
                offset = [x1, y1]
                crop_img = img[y1:y2, x1:x2, :]
                resize_img=cv2.resize(crop_img,(511,511))/255.
                ratio=side/255
                label=np.array([255/ratio,255/ratio,w/ratio,h/ratio]).astype(np.int32)
                label=label.astype(np.float32)

        return resize_img,label,offset,ratio

if __name__=='__main__':
    reader=Image_reader('../data/vot2013/bicycle','../data/vot2013/bicycle/groundtruth.txt')
    pre_box=None
    for i in range(3):
        img,box_ori,img_p,box_p,offset,ratio=reader.get_data(i,pre_box=pre_box)
        if i ==0:
            pre_box=box_ori
        else:
            img_p=(img_p*255).astype(np.uint8)
            # box=reader.recover(box_p,offset,ratio)
            # cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),1)
            cv2.imshow('img',img_p)
            cv2.waitKey(0)

        # box=np.zeros(4)

        # # detection_label_p[2]=detection_label_p[2]*ratio
        # # detection_label_p[3]=detection_label_p[3]*ratio
        # # detection_label_p[0]=detection_label_p[0]*ratio+offset[0]
        # # detection_label_p[1]=detection_label_p[1]*ratio+offset[1]

        # box[0]=int(detection_label_p[0]-(detection_label_p[2]-1)/2)
        # box[1]=int(detection_label_p[1]-(detection_label_p[3]-1)/2)
        # box[2]=int(detection_label_p[0]+(detection_label_p[2]-1)/2)
        # box[3]=int(detection_label_p[1]+(detection_label_p[3]-1)/2)

        # img=(detection_p*255).astype(np.uint8)

        # cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),1)
        # cv2.imshow('img',img)
        # cv2.imshow('img2',(template_p*255).astype(np.uint8))
        # cv2.waitKey(0)





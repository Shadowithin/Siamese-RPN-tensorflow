from net.network import Network

class SiameseRPN(Network):
    def setup(self):
        # (self.feed('template','detection')
        #      #alex net layer 1-5
        #      .conv2(11, 11, 96, 2, 2, padding='VALID', name='conv1')
        #      .lrn2(2, 1.99999994948e-05, 0.75, name='norm1')
        #      .max_pool2(3, 3, 2, 2, padding='VALID', name='pool1')
        #      .conv2(5, 5, 256, 1, 1,padding='VALID', group=2, name='conv2')
        #      .lrn2(2, 1.99999994948e-05, 0.75, name='norm2')
        #      .max_pool2(3, 3, 2, 2, padding='VALID', name='pool2')
        #      .conv2(3, 3, 384, 1, 1, padding='VALID',name='conv3')
        #      .conv2(3, 3, 384, 1, 1, padding='VALID',group=2, name='conv4')
        #      .conv2(3, 3, 256, 1, 1, padding='VALID',group=2, name='conv5'))
        # (self.feed('conv5')
        #      #template
        #      .conv1(3, 3, 2*self.k*256,1,1, padding='VALID', name='t_c',index=0)
        #      .reshape(rate=2,name='t_c_k'))
        # (self.feed('conv5')
        #      #template
        #      .conv1(3, 3, 4*self.k*256,1,1, padding='VALID', name='t_r',index=0)
        #      .reshape(rate=4,name='t_r_k'))
        # (self.feed('conv5')
        #      #detection
        #      .conv1(3, 3, 256,1,1, padding='VALID', name='d_c',index=1))
        # (self.feed('conv5')
        #      #detection
        #      .conv1(3, 3, 256,1,1, padding='VALID', name='d_r',index=1))
        # (self.feed('t_c_k','d_c')
        #      .cf_conv(padding='VALID', name='cls'))#[1,17,17,2k]
        # (self.feed('t_r_k','d_r')
        #      .cf_conv(padding='VALID', name='reg'))#[1,17,17,4k]

        (self.feed('template', 'detection')
            #vgg16 net block 1-4
            #.conv_d(3, 3, 64, 1, 1, padding="SAME", name="conv1_0")
            .conv2(3, 3, 64, 1, 1, padding="SAME", name="conv1_1")
            .conv2(3, 3, 64, 1, 1, padding="SAME", name="conv1_2")
            .max_pool2(2, 2, 2, 2, padding='VALID', name="pool1")#63/255
            .conv2(3, 3, 128, 1, 1, padding="SAME", name="conv2_1")
            .conv2(3, 3, 128, 1, 1, padding="SAME", name="conv2_2")
            .max_pool2(2, 2, 2, 2, padding='VALID', name="pool2")#31/127
            .conv2(3, 3, 256, 1, 1, padding="VALID", name="conv3_1")#29/125
            .conv2(3, 3, 256, 1, 1, padding="VALID", name="conv3_2")#27/123
            .conv2(3, 3, 256, 1, 1, padding="VALID", name="conv3_3")#25/121
            .max_pool2(2, 2, 2, 2, padding='VALID', name="pool3")#12/60
            .conv2(3, 3, 512, 1, 1, padding="VALID", name="conv4_1")#10/58
            .conv2(3, 3, 512, 1, 1, padding="VALID", name="conv4_2")#8/56
            .conv2(3, 3, 512, 1, 1, padding="VALID", name="conv4_3"))#6/54

        (self.feed('conv4_3')
             #template
             .conv1(3, 3, 2*self.k*256,1,1, padding='VALID', name='t_c',index=0)
             .reshape(rate=2,name='t_c_k'))
        (self.feed('conv4_3')
             #template
             .conv1(3, 3, 4*self.k*256,1,1, padding='VALID', name='t_r',index=0)
             .reshape(rate=4,name='t_r_k'))
        (self.feed('conv4_3')
             #detection
             .conv1(3, 3, 256,1,1, padding='VALID', name='d_c',index=1))
        (self.feed('conv4_3')
             #detection
             .conv1(3, 3, 256,1,1, padding='VALID', name='d_r',index=1))
        (self.feed('t_c_k','d_c')
             .cf_conv(padding='VALID', name='cls'))#[1,17,17,2k]
        (self.feed('t_r_k','d_r')
             .cf_conv(padding='VALID', name='reg'))#[1,17,17,4k]
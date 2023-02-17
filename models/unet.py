from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose
from tensorflow.keras.layers import Concatenate, BatchNormalization, Dropout, Cropping2D
from tensorflow.keras.regularizers import l2

import xarray as xr
xr.set_options(display_style='text')

import warnings
warnings.simplefilter("ignore")


class Unet:

    def __init__(self, v, train_patches):

        self.train_patches = train_patches
        self.model_architecture = 'unet'

        # params related to input/preproc.
        if self.train_patches == False:
            self.input_dims = 0
            self.output_dims = 0
        else:
            self.input_dims = 32
            self.output_dims = self.input_dims - 8
            self.patch_stride = 12
            self.patch_na = 4 / 8
        self.n_bins = 3
        self.region = 'global'  # 'europe'

        # params for model architecture
        self.filters = 2
        self.apool = True  # choose between average and max pooling, True = average
        self.n_blocks = 3  # 4  # 5
        self.bn = True  # batch normalization
        self.ct_kernel = (3, 3)  # (2, 2)
        self.ct_stride = (2, 2)  # (2, 2)

        # params related to model training
        self.optimizer_str = 'adam'
        self.call_back = True  # should early stopping be used?

        if v == 'tp':
            self.learn_rate = 0.001
            self.decay_rate = 0.005
            self.delayed_early_stop = True
        else:
            self.learn_rate = 1e-4
            self.decay_rate = 0
            self.delayed_early_stop = False

        if self.train_patches == True:
            self.bs = 32
            self.ep = 20
            self.patience = 3  # for callback
            self.start_epoch = 2  # epoch to start with early stopping
        else:  # global unet
            self.bs = 16
            self.ep = 50  # 20
            self.patience = 10  # for callback
            self.start_epoch = 5
            if self.call_back == False:
                self.ep = 30

    def build_model(self,  dg_train_shape):
        inp_imgs = Input(shape=(dg_train_shape[1],
                                dg_train_shape[2],
                                dg_train_shape[3],))  # fcts

        c0 = inp_imgs

        # encoder / contracting path
        p1, c1 = down(c0, self.filters*4, activation='elu', padding='same',  bn=self.bn, apool=self.apool)  # 16
        p2, c2 = down(p1, self.filters*8, activation='elu', padding='same',  bn=self.bn, apool=self.apool)  # 8
        p3, c3 = down(p2, self.filters*16, activation='elu', padding='same',  bn=self.bn, apool=self.apool)  # 4
        p4, c4 = down(p3, self.filters*32, activation='elu', padding='same',  bn=self.bn, apool=self.apool) if (self.n_blocks >= 4) else [p3, c3]
        p5, c5 = down(p4, self.filters*64, activation='elu', padding='same',  bn=self.bn, apool=self.apool) if (self.n_blocks >= 5) else [p4, c4]

        # bottleneck
        cb = Conv2D(self.filters*4*2**self.n_blocks, (3, 3), activation='elu', padding='same')(p5)
        # cb = Dropout(self.dropout_rate)(cb)
        cb = Conv2D(self.filters*4*2**self.n_blocks, (3, 3), activation='elu', padding='same')(cb)
        cb = BatchNormalization()(cb) if self.bn else cb

        # decoder / expanding path
        u5 = up(cb, c5, self.filters*64, self.ct_kernel, self.ct_stride, activation='elu', padding='same',  bn=self.bn) if (self.n_blocks >=5 ) else cb
        u4 = up(u5, c4, self.filters*32, self.ct_kernel, self.ct_stride, activation='elu', padding='same',  bn=self.bn) if (self.n_blocks >=4 ) else cb
        u3 = up(u4, c3, self.filters*16, self.ct_kernel, self.ct_stride, activation='elu', padding='same',  bn=self.bn)
        u2 = up(u3, c2, self.filters*8, self.ct_kernel, self.ct_stride, activation='elu', padding='same',  bn=self.bn)
        u1 = up(u2, c1, self.filters*4, self.ct_kernel, self.ct_stride, activation='elu', padding='same',  bn=False)  # no normalization directly before softmax

        out = Conv2D(3, (1, 1), activation='softmax')(u1)

        # crop to get rid of patch edges
        if self.train_patches == True:
            out = Cropping2D(cropping=((4, 4), (4, 4)))(out)
        else:
            if self.region == 'europe':
                out = Cropping2D(cropping=((8, 8), (8, 8)))(out)
            if self.region == 'global':
                out = Cropping2D(cropping=((8, 8), (4, 3)))(out)

        cnn = Model(inputs=[inp_imgs], outputs=out)

        cnn.summary()

        return cnn


def down(c, filters, activation='elu', padding='same', lamda=0,
         dropout_rate=0, bn=True, apool=True):
    # lamda: l2 regularizer for kernel and bias
    c = Conv2D(filters, (3, 3), activation=activation, padding=padding,
               kernel_regularizer=l2(lamda), bias_regularizer=l2(lamda))(c)
    c = Dropout(dropout_rate)(c)
    c = Conv2D(filters, (3, 3), activation=activation, padding=padding,
               kernel_regularizer=l2(lamda), bias_regularizer=l2(lamda))(c)
    c = BatchNormalization()(c) if bn else c
    p = AveragePooling2D((2, 2))(c) if apool else MaxPooling2D((2, 2))(c)
    return p, c


def up(u, c, filters, ct_kernel, ct_stride, activation='elu',
       padding='same', lamda=0, dropout_rate=0, bn=True):
    u = Conv2DTranspose(filters, ct_kernel, strides=ct_stride, padding=padding,
                        kernel_regularizer=l2(lamda), bias_regularizer=l2(lamda))(u)  # 8x8
    u = Concatenate()([c, u])  # 8x8
    u = Conv2D(filters, (3, 3), activation=activation, padding=padding,
               kernel_regularizer=l2(lamda), bias_regularizer=l2(lamda))(u)  # pad = same 8x8
    u = Dropout(dropout_rate)(u)
    u = Conv2D(filters, (3, 3), activation=activation, padding=padding,
               kernel_regularizer=l2(lamda), bias_regularizer=l2(lamda))(u)  # pad = same 8x8
    u = BatchNormalization()(u) if bn else u
    return u

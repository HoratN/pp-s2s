from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Activation, Dropout
from tensorflow.keras.layers import Reshape, Flatten, Dot, Add

import numpy as np
import xarray as xr
xr.set_options(display_style='text')

import warnings
warnings.simplefilter("ignore")


class StandardCnn:

    def __init__(self, v, model_architecture, weighted_loss=False):

        self.model_architecture = model_architecture
        self.train_patches = True
        self.weighted_loss = weighted_loss

        # params related to input/preproc.
        self.input_dims = 34
        if model_architecture == 'conv_trans':
            self.output_dims = 16
        else:
            self.output_dims = 8  # for basis_func
        self.n_bins = 3
        self.patch_stride = 12  # 6
        self.patch_na = 4 / 8
        self.region = 'global'

        # params for model architecture
        if (self.model_architecture == 'basis_func') & (v == 'tp'):
            self.filters = [4, 8]  # Scheuerer et al. 2020
            self.hidden_nodes = [10]
        else:
            self.filters = [8, 16]
            self.hidden_nodes = [10, 10]
        self.dropout_rate = 0.4
        if model_architecture == 'basis_func':
            self.n_basis = 9
            self.basis_rad = 16
        else:  # transposed convolutional CNN
            self.oneconv = False

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

        self.start_epoch = 2  # epoch to start with early stopping
        self.bs = 64
        self.ep = 20
        self.patience = 3  # for callback

    def build_model(self, dg_train_shape, dg_train_weight_target=None):

        inp_imgs = Input(shape=(dg_train_shape[1],
                                dg_train_shape[2],
                                dg_train_shape[3],))  # fcts

        if self.model_architecture == 'basis_func':
            n_xy = int(self.output_dims ** 2)
            inp_basis = Input(shape=(n_xy, self.n_basis))  # basis
            inp_cl = Input(shape=(n_xy, self.n_bins,))  # climatology

        c = inp_imgs

        # encoder / contracting path
        for f in self.filters:
            c = Conv2D(f, (3, 3), activation='elu')(c)
            c = MaxPooling2D((2, 2))(c)
        x = Flatten()(c)
        x = Dropout(self.dropout_rate)(x)

        # bottleneck
        for h in self.hidden_nodes:
            x = Dense(h, activation='elu')(x)

        # decoder / expanding path
        if self.model_architecture == 'basis_func':
            #  CNN: slightly adapted from Scheuerer et al. (2020)
            x = Dense(self.n_bins * self.n_basis, activation='elu')(x)
            x = Reshape((self.n_bins, self.n_basis))(x)
            z = Dot(axes=2)([inp_basis, x])  # Tensor product with basis functions
            z = Add()([z, inp_cl])  # Add (log) probability anomalies to log climatological probabilities
            z = Reshape((self.output_dims, self.output_dims, self.n_bins))(z)
        elif self.model_architecture == 'conv_trans':
            z = Dense(self.output_dims * self.n_bins, activation='elu')(x)
            z = Reshape((int(np.sqrt(self.output_dims)), int(np.sqrt(self.output_dims)), self.n_bins))(z)
            z = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same')(z)  # out = 8
            z = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same')(z)  # out = 16
            z = Conv2D(3, (1, 1))(z) if (self.oneconv == True) else z

        out = Activation('softmax')(z)

        # matching input for chosen model
        if self.model_architecture == 'basis_func':
            inputs = [inp_imgs, inp_basis, inp_cl]
        else:
            inputs = [inp_imgs]

        if self.weighted_loss == True:
            weight_shape = dg_train_weight_target[0]
            weights = Input(shape=(weight_shape[1], weight_shape[2],))
            target_shape = dg_train_weight_target[1]
            target = Input(shape=(target_shape[1], target_shape[2], target_shape[3],))

            cnn = Model(inputs=[inputs] + [weights, target], outputs=out)

            cnn.target = target
            cnn.weight_mask = weights
            cnn.out = out
        else:
            cnn = Model(inputs=[inputs], outputs=out)

        # cnn.summary()

        return cnn


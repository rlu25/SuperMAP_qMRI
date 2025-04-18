# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:09:19 2022

@author: rliu25
"""

#%%
from os import path
# import tensorflow.compat.v2 as tf
from supermap import Unet, SuperMap, Trainer

# from keras.datasets import mnist
import torch
import numpy as np
# import tensorflow as tf
# import random
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from scipy.io import loadmat as load

import mat73
from utils import normalize_complex, ifft2,fft2


model = Unet(
    dim = 64,
    dim_mults = (1, 1, 1, 1)
)

train_batch_size = 64
test_data = mat73.loadmat('../get_data_for_model/test_0118_rf8_retro.mat')
x_test = test_data['test']
# x_test=normalize_complex(torch.from_numpy(x_test.astype(np.complex64)))
x_test = np.asarray(x_test)
x_test = x_test.astype(np.float32)
x_test = x_test.reshape(1536,5,41,41)

new_test = torch.from_numpy(x_test)



supermap_model = SuperMap(
    model,
    new_test,
    image_size = 41,
    loss_type = 'l1'            # L1 or L2
).cuda()

train_data = mat73.loadmat('../get_data_for_model/train_data_rf8.mat')
x_train = train_data['data']

x_train = np.asarray(x_train)
# x_train=normalize_complex(torch.from_numpy(x_train.astype(np.complex64)))
# x_train=ifft2(x_train)
x_train = x_train.astype(np.float32)
x_train = x_train.reshape(549695,5,41,41)
x_train = torch.from_numpy(x_train)



trainer = Trainer(
    supermap_model,
    x_train,
    train_batch_size = 64,
    train_lr = 8e-5,
    train_num_steps = 200000,         # total training steps 700000
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay 0.995
    amp = False                        # turn on mixed precision
)
# trainer.load(82)
trainer.train()
# sampled_images = diffusion.sample(batch_size=4) #down size
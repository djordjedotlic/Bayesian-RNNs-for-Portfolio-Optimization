import math
import os
import time
import numpy as np

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn

n_steps = 90
#stocks = ['WIKI/XOM', 'WIKI/GS', 'WIKI/MSFT', 'WIKI/JNJ', 'WIKI/ORCL', 'WIKI/NKE', 'WIKI/MCD', 'WIKI/HPQ',
#         'WIKI/JPM', 'WIKI/BAC', 'WIKI/DIS', 'WIKI/WMT', 'WIKI/OXY', 'WIKI/T', 'WIKI/C']
stocks = ['WIKI/XOM']

n_stocks = len(stocks)

context = mx.cpu(0)
#args_data = '../data/nlp/ptb.'
args_model = 'lstm'
#args_emsize = 30 #should be equal to the batch size
args_nhid = 50
args_nlayers = 2
args_lr = 0.001
args_clip = 0.5
args_epochs = 4000
args_batch_size = 300
args_bptt = n_steps
args_dropout = 0.2
args_tied = False
args_cuda = 'store_true'
args_log_interval = 20
args_save = 'model.param'

w0 = [1/n_stocks] * n_stocks

#coding=utf-8-**-
from __future__ import division
from keras.models import Sequential  # 导入神经网络初始化函数
from keras.layers.core import Dense, Activation  # 导入神经网络层函数、激活函数
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import os,sys, random
random.seed(42)
np.random.seed(42)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras import backend as K
from keras.layers import *
from keras.layers import Activation, Input
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras.models import Model
from JoinAttLayer import Attention
from keras.optimizers import *
from util import *
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
import cPickle as pkl
reload(sys)
sys.setdefaultencoding('utf8')

# original model initial location

train_data = pd.read_csv('../data/train.csv', header=None)
test_data = pd.read_csv('../data/test.csv', header=None)
train_data = train_data.append(test_data)
train_datac = pd.read_csv('../data/trainc.csv', header=None)
test_datac = pd.read_csv('../data/testc.csv', header=None)
train_datac = train_datac.append(test_datac)
train_data = pd.concat([train_data, train_datac], axis=1).reset_index(drop=True)
# print train_data.shape
# test_data = pd.concat([test_data,test_datac],axis = 1)
print train_data.shape

# 10fold data
# 10折的效果，尝试过只取训练中最优的模型，线上效果不如10个模型的融合效果；
td = train_data.as_matrix()
p = 0.1  # 设置训练集的比例
N = 1 / p
# correctRateLM = []
# for i in range(int(N)):
for i in range(10):
    print i
    test_data = td[int(len(td) * (p * i)):int(len(td) * (p * (i + 1))), :]
    train_data = np.row_stack((td[:int(len(td) * (p * i)), :], td[int(len(td) * (p * (i + 1))):, :]))
    test_data = pd.DataFrame(test_data)
    train_data = pd.DataFrame(train_data)

# new model initial location
# 模型是双向lstm+attention+dropout，话说双向lstm本身就很强大了
    maxlen = 24
    wordnum = 4482 + 1
    maxlen2 = 48
    charnum = 1237 + 1
    embedsize = 200
    lstmsize = 10

    input1 = Input(shape=(maxlen,))
    input2 = Input(shape=(maxlen,))
    input3 = Input(shape=(5,))
    embed1 = Embedding(wordnum,embedsize)
	# embedding层的目的是将正整数（下标）转换为具有固定大小的向量
	# 如原本是wordnum维度变为embedsize大小。注意：只能作为模型第一层；
	# 如果输入数据不需要词的语义特征语义，简单使用Embedding层就可以得到一个对应的词向量矩阵，
	# 但如果需要语义特征，我们大可把以及训练好的词向量权重直接扔到Embedding层中即可
	# 具体原理就是神经网络训练得到了。。。
    # lstm0 = CuDNNLSTM(lstmsize,return_sequences = True)
    # lstm1 = Bidirectional(CuDNNLSTM(lstmsize))
    # lstm2 = CuDNNLSTM(lstmsize)
    lstm0 = LSTM(lstmsize,return_sequences = True)
    lstm1 = Bidirectional(LSTM(lstmsize))
    lstm2 = LSTM(lstmsize)
    att1 = Attention(10)
    den = Dense(64,activation = 'tanh')

    # att1 = Lambda(lambda x: K.max(x,axis = 1))
    v3 = embed1(input3)
    v1 = embed1(input1)
    v2 = embed1(input2)
    v11 = lstm1(v1)
    v22 = lstm1(v2)
    v1ls = lstm2(lstm0(v1)) # 支路：对embed后经过两层LSTM
    v2ls = lstm2(lstm0(v2))
    v1 = Concatenate(axis=1)([att1(v1),v11]) # embed后一路经过attention，另一路经过双向LSTM，然后合并两路结果
    v2 = Concatenate(axis=1)([att1(v2),v22])

    input1c = Input(shape=(maxlen2,))
    input2c = Input(shape=(maxlen2,))
    embed1c = Embedding(charnum,embedsize)
    # lstm1c = Bidirectional(CuDNNLSTM(6))
    lstm1c = Bidirectional(LSTM(6))
    att1c = Attention(10)
    v1c = embed1(input1c)
    v2c = embed1(input2c)
    v11c = lstm1c(v1c)
    v22c = lstm1c(v2c)
    v1c = Concatenate(axis=1)([att1c(v1c),v11c])
    v2c = Concatenate(axis=1)([att1c(v2c),v22c])

	# 两路结果做数据增强，特征工程，扩充特征维度，避免过拟合
    mul = Multiply()([v1,v2])
    sub = Lambda(lambda x: K.abs(x))(Subtract()([v1,v2]))
    maximum = Maximum()([Multiply()([v1,v1]),Multiply()([v2,v2])])
    mulc = Multiply()([v1c,v2c])
    subc = Lambda(lambda x: K.abs(x))(Subtract()([v1c,v2c]))
    maximumc = Maximum()([Multiply()([v1c,v1c]),Multiply()([v2c,v2c])])
    sub2 = Lambda(lambda x: K.abs(x))(Subtract()([v1ls,v2ls]))
    matchlist = Concatenate(axis=1)([mul,sub,mulc,subc,maximum,maximumc,sub2])
	# dropout层
    matchlist = Dropout(0.05)(matchlist)

	# 两路dense全连接层合并
    matchlist = Concatenate(axis=1)([Dense(32,activation = 'relu')(matchlist),Dense(48,activation = 'sigmoid')(matchlist)])
    # 最后一层全连接层
	res = Dense(1, activation = 'sigmoid')(matchlist)


    model = Model(inputs=[input1, input2, input3, input1c, input2c], outputs=res)
    model.compile(optimizer=Adam(lr = 0.0015,epsilon=0.000001), loss="binary_crossentropy")
    model.summary()

    # 原来的一次随机0.1分割训练测试
    # index = sorted(random.sample(range(train_data.shape[0]), train_data.shape[0] / 10))
    # test_data = train_data.iloc[index, :]
    # train_data.drop(index, axis=0, inplace=True)
    print test_data.shape
    print train_data.shape

    train_y = train_data.iloc[:, 0]
    train_x1 = train_data.iloc[:, 1:1 + maxlen].values
    train_x2 = train_data.iloc[:, 1 + maxlen:1 + 2 * maxlen].values
    train_x3 = train_data.iloc[:, 1 + 2 * maxlen:1 + 2 * maxlen + 5]
    train_x1c = train_data.iloc[:, 1 + 1 + 2 * maxlen + 5:1 + 1 + 2 * maxlen + 5 + maxlen2].values
    train_x2c = train_data.iloc[:, 1 + 1 + 2 * maxlen + 5 + maxlen2:1 + 1 + 2 * maxlen + 5 + 2 * maxlen2].values

    test_y = test_data.iloc[:, 0]
    test_x1 = test_data.iloc[:, 1:1 + maxlen]
    test_x2 = test_data.iloc[:, 1 + maxlen:1 + 2 * maxlen]
    test_x3 = test_data.iloc[:, 1 + 2 * maxlen:1 + 2 * maxlen + 5]
    test_index = test_data.iloc[:, 1 + 2 * maxlen + 5]
    test_x1c = test_data.iloc[:, 1 + 1 + 2 * maxlen + 5:1 + 1 + 2 * maxlen + 5 + maxlen2]
    test_x2c = test_data.iloc[:, 1 + 1 + 2 * maxlen + 5 + maxlen2:1 + 1 + 2 * maxlen + 5 + 2 * maxlen2]

    # train_y = train_y.apply(lambda x:(x-0.5)*2)
    # test_y = test_y.apply(lambda x:(x-0.5)*2)
    # print test_y.value_counts()

    batch_size = 64
    #batch_size = 256

    for j in range(3):
        print j
        if j == 3:
            #batch_size = 256
            batch_size = 256
            model.compile(optimizer=Adagrad(lr=0.003), loss="binary_crossentropy")
        model.fit([train_x1, train_x2, train_x3, train_x1c, train_x2c], train_y, batch_size=batch_size, epochs=1,
                    shuffle=True,
                    validation_data=([test_x1, test_x2, test_x3, test_x1c, test_x2c], test_y), verbose=1)
        y_pred = model.predict([test_x1, test_x2, test_x3, test_x1c, test_x2c], batch_size=batch_size)
        # print y_pred[:10]
        scoredetail = []
        y_true = np.array(test_y)
        score(y_pred, y_true, t=0.3)

    path = os.path.join('./modelsub'+str(i)+'.h5')
    model.save_weights(path, overwrite=True)
    # model.save_weights('./modelsub.h5',overwrite = True)

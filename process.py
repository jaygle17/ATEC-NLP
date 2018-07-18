#coding=utf-8-**-
import pandas as pd

import jieba.posseg as pseg
import re
import numpy as np
from util import *

spliter1 = re.compile(r"([\&\ \!\:\(\)\*\-\"\/\;\#\+\~\{\[\]\}])")

def proline(line):
    array = line.strip().split('\t')
    s1 = array[1].decode('utf8').replace("***","*")
    s2 = array[2].decode('utf8').replace("***","*")
    s1l = [w for w in jieba.cut(s1) if w.strip()]
    s2l = [w for w in jieba.cut(s2) if w.strip()]
    res = []
    for word in s1l:
        res.append(wdic2.get(word,1))
        # if 1 < wdic2.get(word,1) < 6:
        #     print word,wdic2.get(word,1)
    while len(res) < maxlen:
        res.extend([0] * 5)
        # res.extend(res)
    res = res[:maxlen]
    for word in s2l:
        res.append(wdic2.get(word,1))
    while len(res) < 2 * maxlen:
        res.extend([0] * 5)
        # res.extend(res[maxlen:])
    res = res[:2 * maxlen]
    _, simword = bleu(s1,s2,3,wdic2)
    simword = map(lambda x:x.replace(" ",""),simword)
    simwordl = map(lambda x:wdic2[x],simword)
    simwordl.extend([0] * 5)
    return res,s1l,s2l,simword,simwordl[:5]

def prolinec(line):
    array = line.strip().split('\t')
    s1 = array[1].decode('utf8').replace("***","*")
    s2 = array[2].decode('utf8').replace("***","*")
    s1l = s1
    s2l = s2
    res = []
    for char in s1l:
        res.append(chardict2.get(char,1))
    while len(res) < maxlen2:
        res.extend([0] * 5)
        # res.extend(res)
    res = res[:maxlen2]
    for char in s2l:
        res.append(chardict2.get(char,1))
    while len(res) < 2 * maxlen2:
        res.extend([0] * 5)
        # res.extend(res[maxlen2:])
    res = res[:2 * maxlen2]
    return res

import sys
if __name__ == '__main__':
    inpath = sys.argv[1]
    outpath = sys.argv[2]
    # inpath = '../data/atec_nlp_sim_train.csv'
    # outpath = '../data/result.txt'

    print bleu("abcdefg","abcjhedfkaefg")
    wdic = {} # key:相同窗口词，value:词频

    for line in open('./atec_nlp_sim_train_add.csv'):
        if not line:
            continue
        array = line.strip().split('\t')
        s1 = array[1].decode('utf8')
        s2 = array[2].decode('utf8')
        label = array[3]
        _, res = bleu(s1, s2) # 所以res就是两个文本的相同窗口词
        for k in res:
            wdic[k.replace(' ','')] = wdic.get(k.replace(' ',''), 0) + 0.33

    for line in open('./atec_nlp_sim_train.csv'):
        if not line:
            continue
        array = line.strip().split('\t')
        s1 = array[1].decode('utf8')
        s2 = array[2].decode('utf8')
        # label = array[3]
        _, res = bleu(s1, s2)
        for k in res:
            wdic[k.replace(' ','')] = wdic.get(k.replace(' ',''), 0) + 0.33

    f_out = open('./wdict','w')
    for k,v in wdic.items():
        if v > 15: # 所有训练集中，两两文本间统计相同词频超过15词的词和对应词频，放到wdict中
            print >> f_out,k.encode('utf8'),int(v),"n"
    f_out.close()

    # jieba.load_userdict('./wdict')
	
	# chardict是存储所有汉字的统计词频，放到cdict2中
	# wdic存储所有分词的统计词频包括前面相同窗口词，放到wdict2中；而最前面wdict中是仅仅文本相同窗口词存储
    chardict = {}
    for line in open('./atec_nlp_sim_train_add.csv'):
        if not line:
            continue
        array = line.strip().replace(' ','').split('\t')
        s1 = array[1].decode('utf8')
        s2 = array[2].decode('utf8')
        s1l = [w for w in jieba.cut(s1) if w.strip()]
        s2l = [w for w in jieba.cut(s2) if w.strip()]
        for word in s1l:
            wdic[word] = wdic.get(word, 0) + 1
        for word in s2l:
            wdic[word] = wdic.get(word, 0) + 1
        for char in s1:
            chardict[char] = chardict.get(char,0) + 1
        for char in s2:
            chardict[char] = chardict.get(char,0) + 1

    for line in open('./atec_nlp_sim_train.csv'):
        if not line:
            continue
        array = line.strip().replace(' ','').split('\t')
        s1 = array[1].decode('utf8')
        s2 = array[2].decode('utf8')
        s1l = [w for w in jieba.cut(s1) if w.strip()]
        s2l = [w for w in jieba.cut(s2) if w.strip()]
        for word in s1l:
            wdic[word] = wdic.get(word, 0) + 1
        for word in s2l:
            wdic[word] = wdic.get(word, 0) + 1
        for char in s1:
            chardict[char] = chardict.get(char,0) + 1
        for char in s2:
            chardict[char] = chardict.get(char,0) + 1

    wordflag = {}
    wdic2 = {}
    index = 2 + len(wordflag.items()) # 序号，用处不大
    limit = 5
    f_out = open('./wdict2','w')
    for k,v in wdic.items():
        if v > limit:
            wdic2[k] = index
            print >> f_out,k.encode('utf8'),int(v),"n",index
            index += 1
        else:
            wdic2[k] = 1
            l = [w.flag for w in pseg.cut(k)]
            if len(l) == 1:
                wdic2[k] = wordflag.get(l[0], 1)
            # print >> f_out, k.encode('utf8'), int(v), "n", 1
    f_out.close()
    print index
	
	
    chardict2 = {}
    index = 2
    f_out = open('./cdict2','w')
    for k,v in chardict.items():
        if v > limit:
            chardict2[k] = index
            print >> f_out,k.encode('utf8'),int(v),"n",index
            index += 1
        else:
            chardict2[k] = 1
    f_out.close()
    print index

    maxlen = 24
    maxlen2 = 48
    f_out = open('./debugtra','w')
    f_out2 = open('./train.csv','w')
    f_out3 = open('./trainc.csv', 'w')
    jieba.load_userdict('./dict')
    # jieba.load_userdict('./wdict2')
    for line in open('./atec_nlp_sim_train_add.csv'):
        if not line:
            continue
        line = line.replace(' ', '')
        array = line.strip().split('\t')
        res, s1l, s2l, simword, simwordl = proline(line) # 按照词处理
        resc = prolinec(line) # 按照char字符处理
        # 下面生成train,trainc,debugtra文件的原理
        print >> f_out,(",".join(s1l)).encode('utf8'),(",".join(s2l)).encode('utf8'),array[3],"a" + array[0],(",".join(simword)).encode('utf8')
        print >> f_out2,array[3]+","+",".join(map(str,res))+","+",".join(map(str,simwordl)) + "," + "a" + array[0]
        print >> f_out3, ",".join(map(str, resc)) + "," + "a" + array[0]

    f_out.close()
    f_out2.close()
    f_out3.close()

    f_out = open('./debugtest', 'w')
    f_out2 = open('./test.csv', 'w')
    f_out3 = open('./testc.csv', 'w')
    for line in open(inpath):
        if not line:
            continue
        line = line.replace(' ', '')
        array = line.strip().split('\t')
        res, s1l, s2l, simword, simwordl = proline(line)
        resc = prolinec(line)
        print >> f_out, (",".join(s1l)).encode('utf8'), (",".join(s2l)).encode('utf8'), array[0],(",".join(simword)).encode('utf8')
        print >> f_out2, ",".join(map(str, res))+","+",".join(map(str,simwordl)) + "," + array[0]
        print >> f_out3, ",".join(map(str, resc)) + "," + array[0]

	# 一直出错的原因找到了，以后必须写close,否则很容易出现dataframe或者数组的维度奇怪变化
    f_out.close()
    f_out2.close()
    f_out3.close()

    # evaluation
    test_data = pd.read_csv('./test.csv',encoding = "utf-8",header=None)
    test_datac = pd.read_csv('./testc.csv',encoding = "utf-8", header=None)
    print test_data.shape
    print test_datac.shape
    test_data = pd.concat([test_data, test_datac], axis=1).reset_index(drop=True)
    print test_data.shape

    test_x1 = test_data.iloc[:, 0: maxlen]
    test_x2 = test_data.iloc[:, maxlen: 2 * maxlen]
    test_x3 = test_data.iloc[:, 2 * maxlen:2 * maxlen + 5]
    # test_index = test_data.iloc[:, 1 + 2 * maxlen + 5]
    test_x1c = test_data.iloc[:, 1 + 2 * maxlen + 5:1 + 2 * maxlen + 5 + maxlen2]
    test_x2c = test_data.iloc[:, 1 + 2 * maxlen + 5 + maxlen2:1 + 2 * maxlen + 5 + 2 * maxlen2]

    # 模型初始化
    import numpy as np
    import pandas as pd
    import os, sys, random

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

# original model initial location

    """    
    # 加载权值
    # from keras.models import load_weight
    model.load_weights('./modelsub.h5')
    batch_size = 64
    y_pred = model.predict([test_x1,test_x2,test_x3,test_x1c,test_x2c], batch_size=batch_size)

    lineo = 1
    t = 0.3 # or 0.5
    fout = open(outpath, 'w')
    for line in y_pred:
        if line>t:
            fout.write(str(lineo) + '\t' + '1' + '\n')
        else:
            fout.write(str(lineo) + '\t' + '0' + '\n')
        lineo = lineo + 1
    fout.close()
    """ 
    # 加载权值
    y_sum = []
    for i in range(20):
        print i

# new model initial location
        maxlen = 24
        wordnum = 4482 + 1
        maxlen2 = 48
        charnum = 1237 + 1
        embedsize = 200
        lstmsize = 10

        input1 = Input(shape=(maxlen,))
        input2 = Input(shape=(maxlen,))
        input3 = Input(shape=(5,))
        embed1 = Embedding(wordnum, embedsize) # 词向量embeding
        #lstm0 = CuDNNLSTM(lstmsize, return_sequences=True)
        #lstm1 = Bidirectional(CuDNNLSTM(lstmsize))
        #lstm2 = CuDNNLSTM(lstmsize)
        lstm0 = LSTM(lstmsize, return_sequences=True)
        lstm1 = Bidirectional(LSTM(lstmsize))
        lstm2 = LSTM(lstmsize)
        att1 = Attention(10)
        den = Dense(64, activation='tanh')

        # att1 = Lambda(lambda x: K.max(x,axis = 1))
        v3 = embed1(input3)
        v1 = embed1(input1)
        v2 = embed1(input2)
        v11 = lstm1(v1)
        v22 = lstm1(v2)
        v1ls = lstm2(lstm0(v1))
        v2ls = lstm2(lstm0(v2))
        v1 = Concatenate(axis=1)([att1(v1), v11])
        v2 = Concatenate(axis=1)([att1(v2), v22])

        input1c = Input(shape=(maxlen2,))
        input2c = Input(shape=(maxlen2,))
        embed1c = Embedding(charnum, embedsize)
        #lstm1c = Bidirectional(CuDNNLSTM(6))
        lstm1c = Bidirectional(LSTM(6))
        att1c = Attention(10)
        v1c = embed1(input1c)
        v2c = embed1(input2c)
        v11c = lstm1c(v1c)
        v22c = lstm1c(v2c)
        v1c = Concatenate(axis=1)([att1c(v1c), v11c])
        v2c = Concatenate(axis=1)([att1c(v2c), v22c])

        mul = Multiply()([v1, v2])
        sub = Lambda(lambda x: K.abs(x))(Subtract()([v1, v2]))
        maximum = Maximum()([Multiply()([v1, v1]), Multiply()([v2, v2])])
        mulc = Multiply()([v1c, v2c])
        subc = Lambda(lambda x: K.abs(x))(Subtract()([v1c, v2c]))
        maximumc = Maximum()([Multiply()([v1c, v1c]), Multiply()([v2c, v2c])])
        sub2 = Lambda(lambda x: K.abs(x))(Subtract()([v1ls, v2ls]))
        matchlist = Concatenate(axis=1)([mul, sub, mulc, subc, maximum, maximumc, sub2])
        matchlist = Dropout(0.05)(matchlist)

        matchlist = Concatenate(axis=1)(
            [Dense(32, activation='relu')(matchlist), Dense(48, activation='sigmoid')(matchlist)])
        res = Dense(1, activation='sigmoid')(matchlist)

        model = Model(inputs=[input1, input2, input3, input1c, input2c], outputs=res)
        model.compile(optimizer=Adam(lr=0.0015, epsilon=0.000001), loss="binary_crossentropy")
        model.summary()

        # from keras.models import load_weight
        path = os.path.join('./modelsub' + str(i) + '.h5')
        model.load_weights(path)
        # model.load_weights('./modelsub.h5')
        batch_size = 64
        y_pred = model.predict([test_x1, test_x2, test_x3, test_x1c, test_x2c], batch_size=batch_size)
        # y_pred =  y_pred.tolist()
        y_sum.append(y_pred)
        print y_sum[i][0]
    leng = test_data.shape[0]
    print leng
    y_res = [float(0) for i in range(leng)]
    for i in range(leng):
        for j in range(20):
            y_res[i] = y_res[i] + y_sum[j][i]
    # 直接将相似分数列相加，和t*10比较
    for i in range(20):
        print y_res[i]
    y_res = np.array(y_res)
    print y_res.shape

    lineo = 1
    # t = 0.3 # or 0.5
    t = 6
    fout = open(outpath, 'w')
    for line in y_res:
        if line>t:
            fout.write(str(lineo) + '\t' + '1' + '\n')
        else:
            fout.write(str(lineo) + '\t' + '0' + '\n')
        lineo = lineo + 1
    fout.close()
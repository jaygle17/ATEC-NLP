#/usr/bin/env python
#coding=utf-8
import jieba
import jieba.posseg as pseg
import sys,re
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
        res.append(wdic2.get(word,1)) # 得到词频
        # if 1 < wdic2.get(word,1) < 6:
        #     print word,wdic2.get(word,1)
    while len(res) < maxlen:
        res.extend([0] * 5) # extend:添加内容到现有list中, 
		#词向量通过句子分词的词频表示，意思是词向量不够就补0
		#可以理解同一词词频一样，前后顺序也会体现，这种词向量表达还可以；
        # res.extend(res)
    res = res[:maxlen]
    for word in s2l:
        res.append(wdic2.get(word,1))
    while len(res) < 2 * maxlen:
        res.extend([0] * 5)
        # res.extend(res[maxlen:])
    res = res[:2 * maxlen]
    _, simword = bleu(s1,s2,3,wdic2)
    simword = map(lambda x:x.replace(" ",""),simword) # 相同窗口词
    simwordl = map(lambda x:wdic2[x],simword) # 相同窗口词对应词频，可以看出是词向量表示
    simwordl.extend([0] * 5) # 扩充，保持长度固定
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

if __name__ == '__main__':
    # process(sys.argv[1], sys.argv[2])
    print bleu("abcdefg","abcjhedfkaefg")
    wdic = {} # key:相同窗口词，value:词频

    for line in open('../data/atec_nlp_sim_train_add.csv'):
        if not line:
            continue
        array = line.strip().split('\t')
        s1 = array[1].decode('utf8')
        s2 = array[2].decode('utf8')
        label = array[3]
        _, res = bleu(s1, s2) # 所以res就是两个文本的相同窗口词
		# 构建字典wdic内容什么
        for k in res:
            wdic[k.replace(' ','')] = wdic.get(k.replace(' ',''), 0) + 0.33

    for line in open('../data/atec_nlp_sim_train.csv'):
        if not line:
            continue
        array = line.strip().split('\t')
        s1 = array[1].decode('utf8')
        s2 = array[2].decode('utf8')
        label = array[3]
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
    for line in open('../data/atec_nlp_sim_train_add.csv'):
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

    for line in open('../data/atec_nlp_sim_train.csv'):
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
    index = 2 + len(wordflag.items())
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
    f_out2 = open('../data/train.csv','w')
    f_out3 = open('../data/trainc.csv', 'w')
	# 正则化存储信息到train 和 trainc 
    jieba.load_userdict('./dict')
    # jieba.load_userdict('./wdict2')
    for line in open('../data/atec_nlp_sim_train_add.csv'):
        if not line:
            continue
        line = line.replace(' ', '')
        array = line.strip().split('\t')
        res, s1l, s2l, simword, simwordl = proline(line) # 按照词处理
		#如何把一句输入可以得到分词后两句子s1l,s2l，相同窗口词simword，词向量形式两句子res,相同词simwordl
        resc = prolinec(line) # 按照char字符处理
		# 下面生成train,trainc,debugtra文件的原理
        print >> f_out,(",".join(s1l)).encode('utf8'),(",".join(s2l)).encode('utf8'),array[3],"a" + array[0],(",".join(simword)).encode('utf8')
        print >> f_out2,array[3]+","+",".join(map(str,res))+","+",".join(map(str,simwordl)) + "," + "a" + array[0]
        print >> f_out3, ",".join(map(str, resc)) + "," + "a" + array[0]

    f_out = open('./debugtest', 'w')
    f_out2 = open('../data/test.csv', 'w')
    f_out3 = open('../data/testc.csv', 'w')
    for line in open('../data/atec_nlp_sim_train.csv'):
        if not line:
            continue
        line = line.replace(' ', '')
        array = line.strip().split('\t')
        res, s1l, s2l, simword, simwordl = proline(line)
        resc = prolinec(line)
        print >> f_out, (",".join(s1l)).encode('utf8'), (",".join(s2l)).encode('utf8'), array[3], array[0],(",".join(simword)).encode('utf8')
        print >> f_out2, array[3] + "," + ",".join(map(str, res))+","+",".join(map(str,simwordl)) + "," + array[0]
        print >> f_out3, ",".join(map(str, resc)) + "," + array[0]
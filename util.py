#/usr/bin/env python
#coding=utf-8
import jieba
import sys
import numpy as np

def score(y_pred,y_true,t = 0.5):
	# 计算题目要求的评估标准：tp,tn,fp,fn,  准确率，召回率，
    y_pred = np.array(y_pred).reshape(-1)
    y_true = np.array(y_true).reshape(-1)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] > t and y_true[i] == 1:
            tp += 1
        elif  y_pred[i] <= t and y_true[i] != 1:
            tn += 1
        elif  y_pred[i] <= t and y_true[i] == 1:
            fn += 1
        elif  y_pred[i] > t and y_true[i] != 1:
            fp += 1

    print tp,tn,fp,fn
    print tp*1.0/(tp+fp),tp*1.0/(tp+fn)
    print 2.0*tp/(2*tp+fp+fn)


def bleu(c,r,length = 3, worddict = None):  # function to calc the simiarlity of two words
	# length=3是因为选取了2~4三种窗口模式计算相似度求和
    if len(c) == 0 or len(r) == 0:
        return 0.0
    bp = 0
    sumpn = 0.0
    simwordt = []
    simword = []
    for i in range(1, 1 + length): # 窗口是2~4
        rcount = {}
        for j in range(len(r) - i): # 窗口下字符串可选范围
            w = 1
            rcount[" ".join(r[j:j+1+i])] = rcount.get(" ".join(r[j:j+1+i]),0) + w
			# dict.get(key, default=None)
			# 返回指定键的值，如果值不在字典中返回default值
			# 所以 rcount 统计指定窗口字符串的出现次数
        ccount = {}
        ctcount = {}
        for j in range(len(c) - i):
            w = 1
            t1 = " ".join(c[j:j+1+i])
            ccount[t1] = min(ccount.get(t1,0) + w,rcount.get(t1,0))
			# 而ccount则是统计min(指定窗口大小字符串次数，rcout),从而如果c中有而r中没有的则被统计为0；
            if ccount[t1] > 0:# 针对c和r中共有的窗口字符串
                simwordt.append(t1)
            ctcount[t1] = ctcount.get(t1, 0) + w # 这就是类似与上面rcout,不过统计c中次数
        temp = (1.0/length) * sum(map(lambda x:x[1],ccount.items()))*1.0/(sum(map(lambda x:x[1],ctcount.items()))+0.0001)
        # map(lambda x:x[1],s) # 功能：函数操作，对s中每个元素进行lambda操作：x->x[1]本题就是取出字典的value
		# temp也就是计算指定窗口限制下两个短文本的相似度：通过统计相同窗口词的词频计算相似度；
		sumpn += temp # 而sumpn就是累加三种字符串窗口（2~4）下两个短文本相似度
    for word in simwordt:
        subword = False
        for otherword in simwordt:
            if otherword != word and word in otherword:
                subword = True
                break
        if not subword:
            if worddict and worddict.get(word.replace(' ',''),1) == 1:
                continue
            simword.append(word)
    return bp + sumpn,simword # 返回短文本相似度（大小0~1），和两个短文本中相同的窗口词

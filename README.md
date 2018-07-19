# ATEC-NLP
[大赛链接](https://dc.cloud.alipay.com/index#/topic/intro?id=3)

以TechFin为基础的普惠金融，一个重要目标就是给广大用户提供高效和个性化的客户服务体验。以人工智能技术为核心的智能客服在提升用户体验方面扮演了重要角色。人工智能技术帮助客服人员提供更加高效的服务，在某些场合下甚至能直接向用户提供准确和个性化的客户服务。在经济和技术发展日新月异的今天，客服以其普惠的商业价值和研究价值吸引了大量的专家学者，在学术界得到了广泛的研究。智能客服的本质，就是充分理解用户的意图，在知识体系中精准地找到与之相匹配的内容，回答用户问题或提供解决方案。问题相似度计算，是贯穿智能客服离线、在线和运营等几乎所有环节最核心的技术，同时也是自然语言理解中最核心的问题之一，广泛应用于搜索、推荐、对话等领域。在问题相似度计算上的突破，能够促进整个NLP领域的蓬勃发展，推动通用人工智能的大跨步前进，给人类社会带来巨大的经济价值。

一、赛题任务描述

问题相似度计算，即给定客服里用户描述的两句话，用算法来判断是否表示了相同的语义。

示例：

“花呗如何还款” --“花呗怎么还款”：同义问句
“花呗如何还款” -- “我怎么还我的花被呢”：同义问句
“花呗分期后逾期了如何还款”-- “花呗分期后逾期了哪里还款”：非同义问句
对于例子a，比较简单的方法就可以判定同义；对于例子b，包含了错别字、同义词、词序变换等问题，两个句子乍一看并不类似，想正确判断比较有挑战；对于例子c，两句话很类似，仅仅有一处细微的差别 “如何”和“哪里”，就导致语义不一致。

二、数据

本次大赛所有数据均来自蚂蚁金服金融大脑的实际应用场景，赛制分初赛和复赛两个阶段：

初赛阶段

我们提供10万对的标注数据（分批次更新，已更新完毕），作为训练数据，包括同义对和不同义对，可下载。数据集中每一行就是一条样例。格式如下：

行号\t句1\t句2\t标注，举例：1    花呗如何还款        花呗怎么还款        1

行号指当前问题对在训练集中的第几行；
句1和句2分别表示问题句对的两个句子；
标注指当前问题对的同义或不同义标注，同义为1，不同义为0。
评测数据集总共1万条。为保证大赛的公平公正、避免恶意的刷榜行为，该数据集不公开。大家通过提交评测代码和模型的方法完成预测、获取相应的排名。格式如下：

行号\t句1\t句2

初赛阶段，评测数据集会在评测系统一个特定的路径下面，由官方的平台系统调用选手提交的评测工具执行。

复赛阶段

我们将训练数据集的量级会增加到海量。该阶段的数据不提供下载，会以数据表的形式在蚂蚁金服的数巢平台上供选手使用。和初赛阶段类似，数据集包含四个字段，分别是行号、句1、句2和标注。

评测数据集还是1万条，同样以数据表的形式在数巢平台上。该数据集包含三个字段，分别是行号、句1、句2。

三、评测及评估指标

初赛阶段，比赛选手在本地完成模型的训练调优，将评测代码和模型打包后，提交官方测评系统完成预测和排名更新。测评系统为标准Linux环境，内存8G，CPU4核，无网络访问权限。安装有python 2.7、java 8、tensorflow 1.5、jieba 0.39、pytorch 0.4.0、keras 2.1.6、gensim 3.4.0、pandas 0.22.0、sklearn 0.19.1、xgboost 0.71、lightgbm 2.1.1。 提交压缩包解压后，主目录下需包含脚本文件run.sh，该脚本以评测文件作为输入，评测结果作为输出（输出结果只有0和1），输出文件每行格式为“行号\t预测结果”，命令超时时间为30分钟，执行命令如下：

bash run.sh INPUT_PATH OUTPUT_PATH

预测结果为空或总行数不对，评测结果直接判为0。

复赛阶段，选手的模型训练、调优和预测都是在蚂蚁金服的机器学习平台上完成，后台定时运行选手保存的模型。评测以问题对的两句话作为输入，相似度预测结果（0或1）作为输出，同样输出为空则终止评估，评测结果为0。

本赛题评分以F1-score为准，得分相同时，参照accuracy排序。选手预测结果和真实标签进行比对，几个数值的定义先明确一下：

True Positive（TP）意思表示做出同义的判定，而且判定是正确的，TP的数值表示正确的同义判定的个数； 

同理，False Positive（FP）数值表示错误的同义判定的个数；

依此，True Negative（TN）数值表示正确的不同义判定个数；

False Negative（FN）数值表示错误的不同义判定个数。

基于此，我们就可以计算出准确率（precision rate）、召回率（recall rate）和accuracy、F1-score：

precision rate = TP / (TP + FP)

recall rate = TP / (TP + FN)

accuracy = (TP + TN) / (TP + FP + TN + FN)

F1-score = 2 * precision rate * recall rate / (precision rate + recall rate)

四、问题分析

问题本身属于短文本相似度计算问题，虽然是二分类的目标，但是核心还是计算两个短文本的相似度。而短文本相似度计算主要包括两大类方法：传统机器学习方法，深度学习方法。

由于传统的文本相似性如BM25，无法有效发现语义类 query-Doc 结果对，如"从北京到上海的机票"与"携程网"的相似性、"快递软件"与"菜鸟裹裹"的相似性。在排序时，一些细微的语言变化往往带来巨大的语义变化，如"小宝宝生病怎么办"和"狗宝宝生病怎么办"、"深度学习"和"学习深度"。
因此定位方法为使用深度学习构建学习网络来计算语义相似度。

语句特征提取角度两种不同的模型结构：

  a、sentence encoding-based models that separate the encoding of the individual sentences

  b、joint methods that allow to use encoding of both sentences( to use cross-features or attention from one sentence to the other)
  
第一种比较常见，但是效果不如第二种，比赛中最后使用的是方案b.

五、算法模型

(1)数据预处理函数(pre.py):数据分析，数据增强和词向量模型

  a、数据分析发现20%是正样本, 而80%都是负样本. 因此如果预测所有样本为负样本，那么可以得到 80% acc, 而recall为0%，如果考虑随机数预测结果，发现f1为   0.3左右，所以如果使用模型处理后f1小于0.4那么说明模型没有实际意义。
  
  b、word/char+word embedding得到词向量，经过双向lstm+attention神经网络提取特征后，合并结果，最后经过dropout+dense得到预测模型；
  
  c、数据增强通过对通过神经网络提取特征后的两路输入向量做cross-features，如进行乘法，差模运算，取最大平方，
  
(2)特征工程：

  1)n-gram similiarity(blue score for n-gram=2,3,4);

  2) get length of questions, difference of length

  3) how many words are same, how many words are unique

  4) edit distance

  5) cos similiarity using bag of words for sentence representation(combine tfidf with word embedding from word2vec,fasttext)

  6) manhattan_distance,canberra_distance,minkowski_distance,euclidean_distance
  
(3)DSSM网络训练(dssm.py, JoinAttLayer.py)[1]

考虑在构建模型之前，做分词，词性标注，实体识别，句法分析，语义分析等基础工作；并对数据集进行初步分析和估计；

第一步：通过构建字典，统计词频大小和相似度计算，以及抽取相同词；
第二步：构建字典，存储训练集中所有分词的词频统计，以及汉字的字频统计
第三步：对测试集数据分词后利用词频统计转换为固定长度的词向量，同时分成汉字后利用词频转换为固定长度字向量；
第四步：正则化处理数据等待输入训练和测试；

词向量：固定长度24，利用统计词频方法将短文本分词后构建为词向量表示；
相似度计算：利用窗口截取分词，然后计算 similarity=相同词频/总词频；

DSSM，对于输入数据是Query对，即Query短句和相应的查询短句，查询短句中分相似和不相似，分别为正负样。Word embedding，英文常使用3-grams，对于中文，自己使用了uni-gram，因为中文本身字有一定代表意义（也有论文拆笔画），对于每个gram都使用词频编码代替，最终可以大大降低短句维度[2]。DSSM模型中会加入word hashing层来用letter-trigram向量表示word。

![](https://github.com/jaygle17/ATEC-NLP/blob/master/dssm_theory.png)

具体网络结构如下：双向lstm+attention+dropout+dense，字向量和词向量并行融合，10fold提升性能，最后模型融合达到TOP3%成绩进入复赛；

![](https://github.com/jaygle17/ATEC-NLP/blob/master/dssm_network.png)

其他函数：模型分类预测函数(process.py),工具函数(util.py)

五、比赛tricks

1、相似度计算方法：余弦相似度，欧式距离，
2、使用word和char分别做词向量并融合模型
3、使用n-gram选择n=2，3，4得到相似度，并做平均
4、采用10-fold交叉验证，并融合10个结果比10个中最好的线上结果更好

六、总结思考

还有很多方案没来得及尝试：
1、模型方面：如看到论文用MPCNN模型求解相似度问题；
2、数据方面：准备采用拼音构建词向量融合提升效果
3、tricks方面：可以尝试其他的计算相似度方案，尝试其他词向量训练方法

七、运行环境
python 2.7 + tensorflow 1.8 + keras


[1] https://blog.csdn.net/jaygle/article/details/80927732

[2] https://blog.csdn.net/shine19930820/article/details/79042567



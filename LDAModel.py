#!/usr/bin/python
#coding:utf-8
import numpy as np
import jieba
import lda
from sklearn.feature_extraction.text import CountVectorizer
import ConfigParser
import os

# 当前路径
path = os.getcwd()
# 导入配置文件
conf = ConfigParser.ConfigParser()
conf.read("setting.conf")
# 文件路径
trainfile = os.path.join(path, os.path.normpath(conf.get("filepath", "trainfile")))

# 模型初始化参数
K = int(conf.get("model_args", "K"))
alpha = float(conf.get("model_args","alpha"))
beta = float(conf.get("model_args","beta"))
iter_times = int(conf.get("model_args","iter_times"))
top_words_num = int(conf.get("model_args","top_words_num"))

print trainfile

# 结巴分词并保存到数组
commoditiesNames = []
arr = []
f = open(trainfile)
try:
	line = f.readline()
	while line:
		commoditiesNames.append(line)
		line = jieba.cut(line, cut_all=False)
		arr.append(" ".join(line))
		line = f.readline()
finally:
	f.close()

# 文本特征抽取及向量化
vectorizer = CountVectorizer(min_df=1, analyzer='word', token_pattern='(?u)\\b\\w+\\b')
dtm = vectorizer.fit_transform(arr).toarray()

class LDAModel(object):
	def __init__(self):
		self.K = K
		self.alpha = alpha
		self.beta = beta
		self.iter_times = iter_times
		self.top_words_num = top_words_num
		self.model = lda.LDA(n_topics=self.top_words_num, n_iter=iter_times)
		self.topic_word = None
		self.doc_topic = None

	# 训练模型
	def sampling(self):
		self.model.fit(dtm)
		self.topic_word = self.model.topic_word_
		self.doc_topic = self.model.doc_topic_

	# 模型预测
	def transform(self, query, topK):
		titles = []
		titles.append(" ".join(jieba.cut(query, cut_all=False)))
		dtm_taobao = vectorizer.transform(titles).toarray()
		doc_topic_taobao = self.model.transform(dtm_taobao)
		# 获取预测结果
		result = np.dot(doc_topic_taobao, np.transpose(self.doc_topic))
		# 取得topK
		resTopK = result[0].argsort()[::-1][0:topK]
		print resTopK
		return type(arr)(map(lambda i:commoditiesNames[i], resTopK))	

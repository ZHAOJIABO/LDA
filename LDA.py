import panda as pd
from nlpia.data.loaders import get_data
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
#需要nlpia环境

pd.options.display.width = 120
sms = get_data('sms-spam')
index = ['sms{}{}'.format(i,'!'*j) for (i,j) in zip(range(len(sms)),sms.spam)]
sms = pd.DataFrame(sms.values,columns = sms.columns,index = index)
sms['spam'] = sms.spam.astype(int)
print(len(sms))#总消息数
print(sms.spam.sum())#被标注为垃圾消息的数量
print(sms.head(6))#显示前六条
#所有消息转为tfidf向量
tfidf_model = TfidfVectorizer(tokenizer = casual_tokenize)
tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).tosrray()
print(tfidf_docs.shape)

#计算质心
mask = sms.spam.astype(bool).values
spam_centroid = tfidf_docs[mask].mean(axis=0)
ham_centroid = tfidf_docs[~mask].mean(axis=0)
print(spam_centroid.round(2))
print(ham_centroid.round(2))
#两个质心连线，模型训练得到，定义模型
line = spam_centroid-ham_centroid
#所有tfidf向量投影到line上，计算得分，共完成4837次点积计算
spamminess_score = tfidf_docs.dot(line)
print(spamminess_score.round(2))
#用训练集计算的准确率，没分测试集
sms['lda_score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1,1))
sms['lda_predict'] = (sms.lda_score>0.5).astype(int)
print(sms['spam lda_predict lda_score'.split()].round(2).head(6))
print((1. - (sms.spam - sms.lda_predict).abs().sum()/len(sms)).round(3))







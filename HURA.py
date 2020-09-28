#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import codecs
import numpy as np
 
user_query = []
user_label = []


file = codecs.open('userdata.txt', "r")

for line in file.readlines():
    terms = line.split('\t')
    text = terms[1].lower()
    sentences = text.split('#')

    user_query.append([x.split() for x in sentences])
    user_label.append(int(terms[0]))
file.close()


# In[2]:


word_dict={'PADDING':[0,999999],'UNK':[1,99999]}


for i in user_query:
    for sent in i:
        for word in sent:
            if not word in word_dict:
                word_dict[str(word)]=[len(word_dict),1]
            else:
                word_dict[str(word)][1]+=1


# In[3]:


word_dict_filter={}
for i in word_dict:
    if word_dict[i][1]>=5:
        word_dict_filter[i]=[len(word_dict_filter),word_dict[i][1]]


# In[5]:


embdict=dict()
import  pickle
#all_emb=[]
plo=0
with open('query_word2vec.bin','rb')as f:
    header = f.readline()
    vocab_size, layer1_size = map(int, header.split())
    binary_len = np.dtype('float32').itemsize * layer1_size
    for line in range(vocab_size):
        word = []
        while True:
            ch = f.read(1).decode(errors='ignore')
            if ch ==' ':
                word = ''.join(word)
                break
            if ch != '\n':
                word.append(ch)
        if len(word) != 0:
            tp= np.fromstring(f.read(binary_len), dtype='float32')
            if word in word_dict_filter:
                embdict[word]=tp.tolist()
                if plo%1000==0:
                    print(plo,line,word)
                plo+=1
                #print(word,tp)
        else:
            f.read(binary_len)


# In[6]:


from numpy.linalg import cholesky
lister=[0]*len(word_dict_filter)
xp=np.zeros(200,dtype='float32')

cand=[]
for i in embdict.keys():
    lister[word_dict_filter[i][0]]=np.array(embdict[i],dtype='float32')
    cand.append(lister[word_dict_filter[i][0]])
cand=np.array(cand,dtype='float32')

mu=np.mean(cand, axis=0)
Sigma=np.cov(cand.T)

norm=np.random.multivariate_normal(mu, Sigma, 1)
print(mu.shape,Sigma.shape,norm.shape)

for i in range(len(lister)):
    if type(lister[i])==int:
        lister[i]=np.reshape(norm, 200)
lister[0]=np.zeros(200,dtype='float32')
lister=np.array(lister,dtype='float32')
print(lister.shape)


# In[7]:


maxlen=10
maxsent = 200
user_query_data=[]
for i in user_query:
    userdata=[]
    for sent in i:
        sentence=[]
        for word in sent:
            if word in word_dict:
                 sentence.append(word_dict[str(word)][0])
            if len(sentence)==maxlen:
                break
        userdata.append(sentence+[0]*(maxlen-len(sentence)))
        if len(userdata)==maxsent:
            break
    user_query_data.append(userdata+[[0]*maxlen]*(maxsent-len(userdata)))
 


# In[8]:


qdata = np.array(user_query_data,dtype='int32')
labels = np.array(to_categorical(user_label),dtype='float32')


indices = np.arange(len(qdata))
np.random.shuffle(indices)

nb_validation_samples=int(0.72*len(qdata))
x_train = qdata[indices[:int(0.72*len(qdata))]]
y_train = labels[indices[:int(0.72*len(qdata))]]


x_val = qdata[indices[int(0.72*len(qdata)):int(0.8*len(qdata))]]
y_val = labels[indices[int(0.72*len(qdata)):int(0.8*len(qdata))]]

x_test = qdata[indices[int(0.8*len(qdata)):]]
y_test = labels[indices[int(0.8*len(qdata)):]]


# In[18]:


 


sentence_inputt = Input(shape=(maxlen,), dtype='int32')
embedding_layer = Embedding(len(word_dict_filter), 200, trainable=True)

embedded_sequencest = embedding_layer(sentence_inputt)
word_vec=Dropout(0.2)(embedded_sequencest)
cnn = Convolution1D(nb_filter=300, filter_length=3,  padding='same', activation='relu', strides=1)(word_vec)
d_cnn=Dropout(0.2)(cnn)
w_dense = TimeDistributed(Dense(200,activation='tanh'), name='Dense')(d_cnn)
w_att = Flatten()(Activation('softmax')(Dense(1)(w_dense)))
sent_rep=Dot((1, 1))([d_cnn, w_att])
sentEncodert = Model(sentence_inputt, sent_rep)

userdata_input = Input(shape=(maxsent, maxlen), dtype='int32')
userdata_encoder = TimeDistributed(sentEncodert, name='sentEncodert')(userdata_input)

cnn_sent = Convolution1D(nb_filter=300, filter_length=3, padding='same', activation='relu', strides=1)(userdata_encoder)
d_cnn_sent=Dropout(0.2)(cnn_sent)

s_dense = TimeDistributed(Dense(200,activation='tanh'), name='Dense')(d_cnn_sent)
s_att = Flatten()(Activation('softmax')(Dense(1)(s_dense)))
user_rep=Dot((1, 1))([d_cnn_sent, s_att])

preds = Dense(2, activation='softmax')(user_rep)#age Dense(6, activation='softmax')
model = Model([userdata_input], preds)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
for ep in range(5):
    model.fit([x_train,x_trainq], y_train, epochs=1, batch_size=100)
    y_pred = model.predict([x_val], batch_size=100, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)
    print(classification_report(y_true, y_pred,digits=4))
    print( accuracy_score(y_true, y_pred))


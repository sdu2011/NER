import os,re
import jieba.posseg as pseg
import pycrfsuite
# -*- coding: utf-8 -*-
import sys

#加载处理过的语料(已经标识了每个字的标签 比如如下:)
#浙 B-LOC
#江 I-LOC
#return list [[],[]]
def load_data(filename):
    sents = []
    with open(filename,encoding='utf-8') as f:
        sent = [] #一个完整句子
        for l in f.readlines():
            l = l.replace('\n','')
            character_tmp = l.split(' ')  #['浙', 'B-PRO']
            if len(character_tmp) >= 2: #去除空行或者无标记的行.
                sent.append(character_tmp)
            if character_tmp[0] == '。':
                sents.append(sent)
                sent = []
        return sents

#入参[['浙', 'B-PRO\n'],['江', 'I-PRO\n'],...]
def create_sent(sent):
    one_sent=[]
    for c in sent:
        one_sent.append(c[0])
    return ''.join(one_sent)

#对语句分词,主要为了获得词性,BIO标签体系特征.用以构建特征.
#入参[['浙', 'B-PRO\n'],['江', 'I-PRO\n'],...] --->返回:[['浙', 'B-PRO\n', 'ns', 'B'], ['江', 'I-PRO\n', 'ns', 'I']]
def get_seg(sent):
    new_sent = []
    str_sent = create_sent(sent)
    #print(str_sent)
    c_index = 0 #当前汉字在sent中的index
    words = pseg.cut(str_sent)
    for word, flag in words:
         #print('%s %s %d' % (word, flag,len(word)))
         
         #构建新的特征.词性.字在词中的位置.
         for i in range(len(word)):
            print(c_index)
            
            #字在词中的位置.
            loc = ''
            if len(word) == 1:
                loc = 'S'  #单字成词
            elif i == 0:
                loc = 'B'  #开始
            elif i == len(word) - 1:
                loc = 'E'  #结尾
            else:
                loc = 'I'  #中间
              
            #print('sent',sent)  
            new_sent.append([sent[c_index][0],sent[c_index][1],flag,loc])
            c_index += 1 
            
    return new_sent        

#提取特征,供crf生成特征函数
#get_features([['浙', 'B-PRO\n', 'ns', 'B'], ['江', 'I-PRO\n', 'ns', 'I']])
def get_features(sent):
    sent_features=[]
    for i in range(len(sent)):
        w = sent[i]
        c = w[0]   #当前字
        pos = w[2] #词性,part of speech
        loc = w[3] #字在词中的位置

        features = [
                    'bias',
                    'c='+c,
                    'pos='+pos,
                    'loc='+loc
                   ]

        if i == 0:
            features.append('BOS')
        elif i == len(sent) - 1:
            features.append('EOS')
        else: #考虑上下文.即前一个字和后一个字
            w_pre = sent[i - 1]
            c_pre = w_pre[0]
            pos_pre = w_pre[2]
            loc_pre = w_pre[3]
            
            features.extend(['c_pre='+c_pre,'pos_pre='+pos_pre,'loc='+loc_pre])
            
            w_next = sent[i + 1]
            c_next = w_next[0]
            pos_next = w_next[2]
            loc_next = w_next[3]
            
            features.extend(['c_next='+c_next,'pos_pre='+pos_next,'loc='+loc_next])
            
        sent_features.append(features)
    
    return sent_features        
        
#get_labels([['浙', 'B-PRO\n', 'ns', 'B'], ['江', 'I-PRO\n', 'ns', 'I']])        
def get_labels(sent):
    label_sent = []
    for c in sent:
        label_c = c[1]
        label_sent.append(label_c)
        
    return label_sent
        
    
def from_file_to_features(file):
    #分词,为每一个汉字添加词性信息.构建特征。
    sents = load_data(file)
    all_features = []
    for sent in sents:
        new_sent = get_seg(sent) #添加了词性等信息
        all_features.append(get_features(new_sent))
    
    return all_features
    
def from_file_to_labels(file):
    #分词,为每一个汉字添加词性信息.构建特征。
    sents = load_data(file)
    all_labels = []
    for sent in sents:
        all_labels.append(get_labels(sent))
    
    return all_labels
    
# 训练
def train(X,y,modelname='./model/train.model'):
    trainer = pycrfsuite.Trainer()

    for xseq, yseq in zip(X, y):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    trainer.train(modelname)
    
#[['浙', 'B-PRO\n', 'ns', 'B'], ['江', 'I-PRO\n', 'ns', 'I']]
#文本格式转换  浙江-->[['浙', '', 'ns', 'B'], ['江', '', 'ns', 'I']]
def convert(text):
    features = [] 
    
    c_l = []
    for c in text:
        c_l.append([c,''])
    
    #print(c_l)
    new_sent = get_seg(c_l)

    return new_sent    
    
#识别一个文本.    
def predict(text,modelname='./model/train.model'):
    tagger = pycrfsuite.Tagger()
    tagger.open(modelname)
    
    text_c = convert(text)
    features = get_features(text_c)
    
    print(features)
    tag_result=tagger.tag(features)
    
    return tag_result

#train(all_features,all_labels)
        
        
if __name__ == '__main__':
    print("usage:python main.py [method] [param]")
    print("usage:python main.py train train_file [modelname]")
    print("usage:python main.py predict text [modelname]")
    if sys.argv[1] == 'train':
        file = sys.argv[2]
        features=from_file_to_features(file)
        labels=from_file_to_labels(file)
        train(features,labels)
    elif sys.argv[1] == 'predict':
        text = sys.argv[2]
        print(predict(text))
        
    

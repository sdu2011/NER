# NER
中文NER

使用公开语料,基于条件随机场的中文NER.  
使用说明:  
对某个句子进行ner:  python main.py predict 李华2020年毕业,毕业后想去杭州工作  
训练模型：python main.py train train_file [modelname]  
评价模型：python main.py measure [modelname]

__author__ = "Luke Liu"
#encoding="utf-8"
"""
@Author:Che_Hongshu
@Modify:2018.1.8
"""
import codecs
import os

DATA_TYPE = "english"  # 将DATA_TYPE先后设置为chinese,english得到中英文ＶＯＣＡＢ文件
train_dataset_path='D:/BaiduYunDownload/python_exe/dataset/en-zh'

if DATA_TYPE == "chinese":  # 翻译语料的中文部分
    RAW_DATA = os.path.join(train_dataset_path,"train.txt.zh")
    VOCAB = "vocab_dicts/zh.vocab"
    OUTPUT_DATA = "preprocessed_data/train.zh"
elif DATA_TYPE == "english":  # 翻译语料的英文部分
    RAW_DATA = os.path.join(train_dataset_path,"train.txt.en")
    VOCAB = "vocab_dicts/en.vocab"
    OUTPUT_DATA = "preprocessed_data/train.en"

with codecs.open(VOCAB, 'r', 'utf-8') as f_vocab:  #打开文件进入读操作
    vocab = [w.strip() for w in f_vocab.readlines()]  # 先把所有词转换成list
    # 把每个词和所在行数对应起来并且zip打包成(“词”，行数)格式转换成dict格式

#建立字典
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

# 返回id 如果在词汇表文件中则返回对应的id即可，如果没有则返回'<unk>'
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id['<unk>']

# 打开文件
fin = codecs.open(RAW_DATA, 'r', 'utf-8')
fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')

for line in fin:
    # 每一行的单词变成sring list格式，每一句话后面加上一个结束符号
    words = line.strip().split() + ["<eos>"]
    # 这一行中的每个单词取出对应的id之后用空格相连接
    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
    fout.write(out_line)

# 关闭文件
fin.close()
fout.close()

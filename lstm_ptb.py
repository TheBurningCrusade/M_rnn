# pylint:skip-file
#encoding:utf-8
import lstm
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np

"""
PennTreeBank Language Model
We would like to thanks Wojciech Zaremba for his Torch LSTM code

The data file can be found at:
https://github.com/dmlc/web-data/tree/master/mxnet/ptb
"""
# 这里我们加了一个token变量，用来测试train/valid使用的dic，本程序必须保证valid中的
# 出现的所有的word都必须出现在train数据
def load_data(path, token, dic=None,):
    fi = open(path)
    content = fi.read()
    # print type(content)
    content = content.replace('\n', '<eos>')
    # print type(content)
    # 前两部操作content都是str，现在content为list
    content = content.split(' ')
    # print type(content)
    print("Loading %s, size of data = %d" % (path, len(content)))
    # 生成一个和content一样长的x，x为一个list，每一个元素都是一个float，初始值0.0
    x = np.zeros(len(content))
    if dic == None:
        dic = {}
        # 载入验证数据时dic是用train生成的dic，保证字典统一
        # print >> sys.stdout, 'test valid'
    idx = 0
    # for循环实现两个功能，1.使用dic对content总的word进行编号;2.将content借助dic
    # 进行转化，其中下标x[i]，i和content一一对应word的位置，而x[i]的值代表的是word
    # 在dic中的编号
    for i in range(len(content)):
        word = content[i]
        if len(word) == 0:
            continue
        if not word in dic:
            dic[word] = idx
            idx += 1
            # 测试dic的编号
            if token == 'valid':
                print >> sys.stdout, '%s' % word
        x[i] = dic[word]
    print("Unique token: %d" % len(dic))
    return x, dic

def drop_tail(X, seq_len):
    shape = X.shape
    # 得到的行数x.shape[0]在这个函数中还有进行按seq_len进行取模
    # 使返回的X的行数可以被seq_len整除
    # print >> sys.stdout, 'drop tail:shape[0]: %d' % shape[0]
    nstep = int(shape[0] / seq_len)
    # print 'ddd'
    # print X[0:(nstep * seq_len), :].shape[0]
    # print nstep
    return X[0:(nstep * seq_len), :]


def replicate_data(x, batch_size):
    # print >> sys.stdout, 'batch_size: %d' % batch_size
    # print >> sys.stdout, 'x.shape[0]: %d' % x.shape[0]
    # x是numpy中array类型, x 是1维的, shape暂时代表x的长度
    # 这里x和x_cut的区别是x的长度不能被batch_size整除整除,而x_cut可以被batch_size整除
    nbatch = int(x.shape[0] / batch_size)
    #print >> sys.stdout, 'nbatch: %d' % (nbatch)
    x_cut = x[:nbatch * batch_size]
    # print >> sys.stdout, 'x_cut size: %d' % x_cut.shape[0]
    # 这里将x_cut转换成一个二维矩阵, 矩阵有nbatch行，batch_sie列
    data = x_cut.reshape((nbatch, batch_size), order='F')
    return data

batch_size = 20
seq_len = 35
num_hidden = 200
num_embed = 200
num_lstm_layer = 2
num_round = 25
learning_rate= 0.1
wd=0.
momentum=0.0
max_grad_norm = 5.0
update_period = 1


X_train, dic = load_data("./data/ptb.train.txt", 'train')
#print "X_train first phase"
# print X_train[0:21] # [  0.   0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  
# 11.  12.  13. 14.  15.  16.  17.  18.  19.]
X_val, _ = load_data("./data/ptb.valid.txt", 'valid', dic)
X_train_batch = replicate_data(X_train, batch_size)
X_val_batch = replicate_data(X_val, batch_size)
#print "shape X"
#print X_train_batch.shape
vocab = len(dic)
print("Vocab=%d" %vocab)

X_train_batch = drop_tail(X_train_batch, seq_len)
X_val_batch = drop_tail(X_val_batch, seq_len)
#print 'ddddd'
print X_train_batch.shape
print X_val_batch.shape



model = lstm.setup_rnn_model(mx.cpu(),
                             num_lstm_layer=num_lstm_layer,
                             seq_len=seq_len,
                             num_hidden=num_hidden,
                             num_embed=num_embed,
                             num_label=vocab,
                             batch_size=batch_size,
                             input_size=vocab,
                             initializer=mx.initializer.Uniform(0.1),dropout=0.5)
# max_grad_norm=5.0 | update_period=1 | wd=0 | learning_rate=0.1 | num_roud=25
lstm.train_lstm(model, X_train_batch, X_val_batch,
                num_round=num_round,
                half_life=2,
                max_grad_norm = max_grad_norm,
                update_period=update_period,
                learning_rate=learning_rate,
                wd=wd)
#               momentum=momentum)


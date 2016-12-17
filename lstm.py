# pylint:skip-file
#encoding:utf-8
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    '''这里的i2h和h2h是由裸的indata和prev_stata.h经过FullyConnected得到的,
    表示裸的indata和prev_state.h和他们的权重矩阵相乘，得到每个隐藏单元的
    输入值,这些输入值是隐藏单元激活函数的输入,这里的激活函数不是一个单一的
    函数，而是由忘记门，输入没，输出门构成的一个block，而整个bolck则替代
    隐藏单元激活函数。
    要注意这里设置的隐藏单元的个数是正常的4倍，这就表示一个block中的忘记门，
    输入门，输入转化，输入门都使用了自己单独的权重矩阵和裸(indata和prev_state.h)
    进行相乘运算，从而得到不同的输入数据作为各自门的输入
    '''
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    #print type(i2h)
    #print type(h2h)
    '''这里的相加表示的是i2h和h2h中的矩阵进行相加操作，本例中seq_num为35（它应该对应rnn
    中的时刻的长度，即展开的rnn中一共由35层，每一层对应一个时刻），batch_size为20,它应该
    是每一个时刻数据数据的基本长度，每一个代表一个单词。
    本程序中每一个时刻indata和prev_state.h都是一个20X200的矩阵，lstm中隐藏层单元的个数设
    定为200*4=800个，那么这里的fullyconnectd就是20X200 * 200X800的一个运算
    '''
    gates = i2h + h2h
    # print type(gates)
    # print dir(i2h)
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                     name="t%d_l%d_slice" % (seqidx, layeridx))
    # print type(slice_gates)
    # print type(slice_gates[0])
    # 这是个输入门，上一个时刻的输出和此时的输入作为sigmoid函数的输入，它用来决定
    # 应该更新的值
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    # print in_gate
    # 这里的in_transform用来生成新的候选值，in_gate和in_transform的结果会进行相乘
    # 运算,应为sigmoid函数输出的是0 or 1 所以，他两个相乘是用来表示新生成的候选值
    # 是否要会进入到后面的更新中
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    # 这个是忘记门，有上一个时刻的输出和此时的输入作为sigmoid函数的输入，sigmoid函数
    # 的输出还要和上一个时刻的状态相乘，一个sigmoid函数和一个相乘运算构成了一个门,但
    # 这里是一个sigmoid运算就将他指定为一个门
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    # 这个是输出门，它也是使用一个时刻的输出和此时的输入作为sigmoid函数的输入
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    # 此时细胞的状态
    # print prev_state.c
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    # 此时细胞的输出
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    # print dir(next_h)
    # print next_h
    return LSTMState(c=next_c, h=next_h)


def lstm_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_embed, num_label, dropout=0.):
    """unrolled lstm network"""
    # initialize the parameter symbols
    embed_weight=mx.sym.Variable("embed_weight")
    #print type(embed_weight)
    cls_weight = mx.sym.Variable("cls_weight")
    #print type(cls_weight)
    cls_bias = mx.sym.Variable("cls_bias")
    #print type(cls_bias)
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        # param_cells是两层lstm中的weigh和bias，他们之间是使用fullconnected连接的
        param_cells.append(LSTMParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                      i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                      h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                      h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)))
        # last_states中是seq_num 中每个lstm单元前一个时间的状态(c)和输出(h)
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    label = mx.sym.Variable("label")
    #print label
    last_hidden = []
    #print 'seq——len'
    #print seq_len # 35
    for seqidx in range(seq_len):
        # embeding layer
        data = mx.sym.Variable("t%d_data" % seqidx)
        # print data.__dict__
        #print 'input_size'
        #print input_size
        #print 'num_embedding'
        #print num_embed
        # print dir(data)
        # input_size = 10000  num_embed = 200
        hidden = mx.sym.Embedding(data=data, weight=embed_weight,
                                  input_dim=input_size,
                                  output_dim=num_embed,
                                  name="t%d_embed" % seqidx)
        # print hidden.__dict__
        # print dir(hidden)
        # stack LSTM
        for i in range(num_lstm_layer):
            if i==0:
                dp=0.
            else:
                dp = dropout
            # 对于seq_num， 他的for循环, lstm只有indata和seqidx这两个参数不一样
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp)
            # 使用lstm处理后的next_state和next_state.h会接着作为下一个
            # seq的输入的, 其中hidden只保留每个seq_idx的最后一层的的
            # 数据，而last_states保留当前所有lstm层的next_states结果，并
            # 作为下一个seq_idx的输入
            hidden = next_state.h
            # print dir(hidden)
            # last_states[i]的值会作为后续的seq使用
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        last_hidden.append(hidden)
    # 在python中函数定义中传入一个星号的将会把传入的参数转换成tuple，而
    # 传入两个星号的表明将传入的参数转化成字典比如a=1,b=2 转化成{'a':1, 'b':2}
    concat = mx.sym.Concat(*last_hidden, dim = 0)
    # print type(concat)
    # 将生成的的35个hidden，连接起来，并将他们和num_label个异常单元进行fully
    # connected, 即矩阵相乘，这里每一个hidden都是一个200维的向量
    #print 'num_label'
    #print num_label
    # 这里的最后一层的隐藏层有10000个，它的权重矩阵是10000X200
    # 表示的是最后的隐藏层到输出层之后的矩阵相乘
    fc = mx.sym.FullyConnected(data=concat,
                               weight=cls_weight,
                               bias=cls_bias,
                               num_hidden=num_label)
    # 对每个隐藏单元进行softmax运算,这样可以将每个隐藏单元的输出规则化成
    # 概率值,该概率值即表示一个label的概率,因为本来隐藏单元的个数和label的
    # 个数相等
    #print 'label'
    #print label
    sm = mx.sym.SoftmaxOutput(data=fc, label=label, name="sm")
    # print dir(sm)
    out_prob = [sm]
    # print out_prob[0]
    # print type(out_prob)
    for i in range(num_lstm_layer):
        state = last_states[i]
        state = LSTMState(c=mx.sym.BlockGrad(state.c, name="l%d_last_c" % i),
                          h=mx.sym.BlockGrad(state.h, name="l%d_last_h" % i))
        last_states[i] = state

    unpack_c = [state.c for state in last_states]
    #print 'unpack_c'
    #print len(unpack_c)
    unpack_h = [state.h for state in last_states]
    list_all = out_prob + unpack_c + unpack_h
    # print len(list_all)
    # print type(list_all)
    return mx.sym.Group(list_all)


def is_param_name(name):
    return name.endswith("weight") or name.endswith("bias") or\
        name.endswith("gamma") or name.endswith("beta")


def setup_rnn_model(ctx,
                    num_lstm_layer, seq_len, # seq_len=35, num_lstm_layer=2
                    num_hidden, num_embed, num_label, # num_hidden=200, num_embed=200,
                    batch_size, input_size,  # batch_size=20
                    initializer, dropout=0.):
    print 'num_label'
    print num_label
    """set up rnn model with lstm cells"""
    rnn_sym = lstm_unroll(num_lstm_layer=num_lstm_layer,
                          num_hidden=num_hidden,
                          seq_len=seq_len,
                          input_size=input_size,
                          num_embed=num_embed,
                          num_label=num_label,
                          dropout=dropout)
    # print dir(rnn_sym)
    # print type(rnn_sym)
    arg_names = rnn_sym.list_arguments()
    # last_state中的c和h都是存储的是经过2层lstm处理之后的结果，因为lstm在
    # 每层的个数都是200个隐藏单元,每个隐藏单元的输出都是一个输出的一个数值
    output_names = rnn_sym.list_outputs()
    print 'output_names'
    print output_names
    print 'arg_names'
    print arg_names
    print len(arg_names)

    input_shapes = {}
    for name in arg_names:
        if name.endswith("init_c") or name.endswith("init_h"):
            # 这里制定了第一时刻时，需要需要以来的prev_state, 设定是一个20X200的矩阵
            input_shapes[name] = (batch_size, num_hidden)
        elif name.endswith("data"):
            input_shapes[name] = (batch_size, )
        else:
            pass
    # input_shapes除了有seq_num(35)个输入数据，还有４个0(1)_init_c和0(1)_init_h
    # 一共39个参数
    print 'input_shapes'
    print len(input_shapes)
    print input_shapes
    #for i in range(len(input_shapes)):
    #    print >> '%d : %s' % (i, input_shapes[i])
    #print len(input_shapes)
    '''arg_shape 和 arg_names是对应的，其中arg_names以list的形式存储rnn中的
    所有参数; arg_shape则以list存储对应arg_names参数的维度; arg_array也和
    arg_shape 和　arg_names对应，它也是list类型，其中每个元素是mx.nd，是
    每个参数的的存储空间'''
    arg_shape, out_shape, aux_shape = rnn_sym.infer_shape(**input_shapes)
    print 'arg_shape'
    #for i in range(len(arg_shape)):
    #    print >> '%d'  % i
    #    print arg_shape[i]
    print arg_shape
    print len(arg_shape)
    print 'out_shape'
    print out_shape
    print 'aux_shape'
    print aux_shape
    #print 'ctx'
    #print ctx
    #print type(ctx)
    arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
    #print 'arg_arrays'
    #print arg_arrays
    #print len(arg_arrays)
    # args_grad是每个参数对应的求偏导的结果，它和arg_array拥有一样的维度
    args_grad = {}
    for shape, name in zip(arg_shape, arg_names):
        if is_param_name(name):
            args_grad[name] = mx.nd.zeros(shape, ctx)
    #print 'args_grad'
    #print args_grad
    #print len(args_grad)

    rnn_exec = rnn_sym.bind(ctx=ctx, args=arg_arrays,
                            args_grad=args_grad,
                            grad_req="add")
    param_blocks = []
    arg_dict = dict(zip(arg_names, rnn_exec.arg_arrays))
    #print 'arg_dict'
    #print arg_dict
    for i, name in enumerate(arg_names):
        if is_param_name(name):
            initializer(name, arg_dict[name])

            param_blocks.append((i, arg_dict[name], args_grad[name], name))
        else:
            assert name not in args_grad


    print 'param_blocks'
    print param_blocks
    for i in range(len(param_blocks)):
        print >> sys.stdout, '%d\t%s' % (param_blocks[i][0], param_blocks[i][3])
    print len(param_blocks)


    out_dict = dict(zip(rnn_sym.list_outputs(), rnn_exec.outputs))
    print "out_dict"
    print out_dict

    init_states = [LSTMState(c=arg_dict["l%d_init_c" % i],
                             h=arg_dict["l%d_init_h" % i]) for i in range(num_lstm_layer)]
    seq_labels = rnn_exec.arg_dict["label"]
    seq_data = [rnn_exec.arg_dict["t%d_data" % i] for i in range(seq_len)]
    last_states = [LSTMState(c=out_dict["l%d_last_c_output" % i],
                             h=out_dict["l%d_last_h_output" % i]) for i in range(num_lstm_layer)]
    seq_outputs = out_dict["sm_output"]

    return LSTMModel(rnn_exec=rnn_exec, symbol=rnn_sym,
                     init_states=init_states, last_states=last_states,
                     seq_data=seq_data, seq_labels=seq_labels, seq_outputs=seq_outputs,
                     param_blocks=param_blocks)



def set_rnn_inputs(m, X, begin):
    '''这里是为epoch即一个seq_num=35设置输入输出，输入都是一个20维的向量即为
    X的列数（即X的一行），输出为输入所在行的下一行，也是20维的向量.一行数据
    既作为该轮（seqidx)的输入，同时又作为上一轮(seqidx - 1)的输出。这样一个
    epoch就会又35 X 20 = 700的输出向量(seq_labels)
    '''
    seq_len = len(m.seq_data)
    batch_size = m.seq_data[0].shape[0]
    for seqidx in range(seq_len):
        idx = (begin + seqidx) % X.shape[0]
        next_idx = (begin + seqidx + 1) % X.shape[0]
        x = X[idx, :]
        y = X[next_idx, :]
        #print "x"
        #print x
        #print "y"
        #print y
        '''print 'idx'
        print idx
        print 'next_idx'
        print next_idx
        print 'x'
        print len(x)
        print 'y'
        print len(y)'''
        mx.nd.array(x).copyto(m.seq_data[seqidx])
        m.seq_labels[seqidx*batch_size : seqidx*batch_size+batch_size] = y
    #print 'xxxxxxxxxxxxxx'

def calc_nll(seq_label_probs, X, begin):
    nll = -np.sum(np.log(seq_label_probs.asnumpy())) / len(X[0,:])
    return nll

#max_grad_norm=5.0 | update_period=1 | wd=0 | learning_rate=0.1 | num_roud=25 
def train_lstm(model, X_train_batch, X_val_batch,
               num_round, update_period,
               optimizer='sgd', half_life=2,max_grad_norm = 5.0, **kwargs):
    print("Training swith train.shape=%s" % str(X_train_batch.shape))
    print("Training swith val.shape=%s" % str(X_val_batch.shape))
    print "first row data"
    print X_train_batch[0]
    m = model
    seq_len = len(m.seq_data)
    batch_size = m.seq_data[0].shape[0]
    print("batch_size=%d" % batch_size)
    print("seq_len=%d" % seq_len)

    opt = mx.optimizer.create(optimizer,
                              **kwargs)
    print 'opt'
    print type(opt)

    updater = mx.optimizer.get_updater(opt)
    print 'updater'
    print type(updater)
    epoch_counter = 0
    log_period = max(1000 / seq_len, 1)
    last_perp = 10000000.0
    
    for iteration in range(1):
        nbatch = 0
        train_nll = 0
        # reset states
        for state in m.init_states:
            state.c[:] = 0.0
            state.h[:] = 0.0
        tic = time.time()
        assert X_train_batch.shape[0] % seq_len == 0
        assert X_val_batch.shape[0] % seq_len == 0
        for begin in range(0, X_train_batch.shape[0], seq_len):
            set_rnn_inputs(m, X_train_batch, begin=begin)
            m.rnn_exec.forward(is_train=True)
            # probability of each label class, used to evaluate nll
            print 'm.seq_outputs'
            #print type(m.seq_outputs)
            print m.seq_outputs.shape
            #print m.seq_outputs.asnumpy()
            print 'm.seq_labes'
            #print m.seq_labels.asnumpy()
            print m.seq_labels.shape
            '''mx.nd.choos_element_0index这个的第一个参数是一个矩阵，在这里是一个2维
            的；第二个参数是一个一维向量，向量的长度和第一个参数的行数是一样长的，
            向量中的每一个值都是对应行的index。函数的功能是第二个参数的下标为第一个
            参数的行的索引下标，第二个参数的下标对应的值为第一个参数在该行的列索引下
            标,这样就会取出一个向量'''
            seq_label_probs = mx.nd.choose_element_0index(m.seq_outputs,m.seq_labels)
            m.rnn_exec.backward()
            # transfer the states
            # 将前面的seq_num(35)个计算的到的last_state，作为下一个时刻(seq_num
            # =35)的输入，这样就相当于整个rnn的展开层是无限增加的
            for init, last in zip(m.init_states, m.last_states):
                last.c.copyto(init.c)
                last.h.copyto(init.h)
            # update epoch counter
            epoch_counter += 1
            if epoch_counter % update_period == 0:
                # updare parameters
                norm = 0.
                for idx, weight, grad, name in m.param_blocks:
                    grad /= batch_size
                    l2_norm = mx.nd.norm(grad).asscalar()
                    norm += l2_norm*l2_norm
                norm = math.sqrt(norm)
                for idx, weight, grad, name in m.param_blocks:
                    if norm > max_grad_norm:
                        grad *= (max_grad_norm / norm)
                    updater(idx, grad, weight)
                    # reset gradient to zero
                    grad[:] = 0.0
            train_nll += calc_nll(seq_label_probs, X_train_batch, begin=begin)

            nbatch = begin + seq_len
            if epoch_counter % log_period == 0:
                print("Epoch [%d] Train: NLL=%.3f, Perp=%.3f" % (
                    epoch_counter, train_nll / nbatch, np.exp(train_nll / nbatch)))
        # end of training loop
        toc = time.time()
        print("Iter [%d] Train: Time: %.3f sec, NLL=%.3f, Perp=%.3f" % (
            iteration, toc - tic, train_nll / nbatch, np.exp(train_nll / nbatch)))
'''
        val_nll = 0.0
        # validation set, reset states
        for state in m.init_states:
            state.c[:] = 0.0
            state.h[:] = 0.0
        for begin in range(0, X_val_batch.shape[0], seq_len):
            set_rnn_inputs(m, X_val_batch, begin=begin)
            m.rnn_exec.forward(is_train=False)
            # probability of each label class, used to evaluate nll
            seq_label_probs = mx.nd.choose_element_0index(m.seq_outputs,m.seq_labels)
            # transfer the states
            for init, last in zip(m.init_states, m.last_states):
                last.c.copyto(init.c)
                last.h.copyto(init.h)
            val_nll += calc_nll(seq_label_probs, X_val_batch, begin=begin)
        nbatch = X_val_batch.shape[0]
        perp = np.exp(val_nll / nbatch)
        print("Iter [%d] Val: NLL=%.3f, Perp=%.3f" % (
            iteration, val_nll / nbatch, np.exp(val_nll / nbatch)))
        if last_perp - 1.0 < perp:
            opt.lr *= 0.5
            print("Reset learning rate to %g" % opt.lr)
        last_perp = perp
'''

def setup_rnn_sample_model(ctx,
                           params,
                           num_lstm_layer,
                           num_hidden, num_embed, num_label,
                           batch_size, input_size):
    seq_len = 1
    rnn_sym = lstm_unroll(num_lstm_layer=num_lstm_layer,
                          num_hidden=num_hidden,
                          seq_len=seq_len,
                          num_embed=num_embed,
                          num_label=num_label)
    arg_names = rnn_sym.list_arguments()
    input_shapes = {}
    for name in arg_names:
        if name.endswith("init_c") or name.endswith("init_h"):
            input_shapes[name] = (batch_size, num_hidden)
        elif name.endswith("data"):
            input_shapes[name] = (batch_size, input_size)
        else:
            pass
    arg_shape, out_shape, aux_shape = rnn_sym.infer_shape(**input_shapes)
    arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
    arg_dict = dict(zip(arg_names, arg_arrays))
    for name, arr in params.items():
        arg_dict[name][:] = arr
    rnn_exec = rnn_sym.bind(ctx=ctx, args=arg_arrays, args_grad=None, grad_req="null")
    out_dict = dict(zip(rnn_sym.list_outputs(), rnn_exec.outputs))
    param_blocks = []
    params_array = list(params.items())
    for i in range(len(params)):
        param_blocks.append((i, params_array[i][1], None, params_array[i][0]))
    init_states = [LSTMState(c=arg_dict["l%d_init_c" % i],
                             h=arg_dict["l%d_init_h" % i]) for i in range(num_lstm_layer)]
    seq_labels = [rnn_exec.arg_dict["t%d_label" % i] for i in range(seq_len)]
    seq_data = [rnn_exec.arg_dict["t%d_data" % i] for i in range(seq_len)]
    last_states = [LSTMState(c=out_dict["l%d_last_c_output" % i],
                             h=out_dict["l%d_last_h_output" % i]) for i in range(num_lstm_layer)]
    seq_outputs = [out_dict["t%d_sm_output" % i] for i in range(seq_len)]

    return LSTMModel(rnn_exec=rnn_exec, symbol=rnn_sym,
                     init_states=init_states, last_states=last_states,
                     seq_data=seq_data, seq_labels=seq_labels, seq_outputs=seq_outputs,
                     param_blocks=param_blocks)

# Python3 np.random.choice is too strict in eval float probability so we use an alternative
import random
import bisect
import collections

def _cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def _choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = _cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]

def sample_lstm(model, X_input_batch, seq_len, temperature=1., sample=True):
    m = model
    vocab = m.seq_outputs[0].shape[1]
    batch_size = m.seq_data[0].shape[0]
    outputs_ndarray = mx.nd.zeros(m.seq_outputs[0].shape)
    outputs_batch = []
    tmp = [i for i in range(vocab)]
    for i in range(seq_len):
        outputs_batch.append(np.zeros(X_input_batch.shape))
    for i in range(seq_len):
        set_rnn_inputs(m, X_input_batch, 0)
        m.rnn_exec.forward(is_train=False)
        outputs_ndarray[:] = m.seq_outputs[0]
        for init, last in zip(m.init_states, m.last_states):
            last.c.copyto(init.c)
            last.h.copyto(init.h)
        prob = np.clip(outputs_ndarray.asnumpy(), 1e-6, 1 - 1e-6)
        if sample:
            rescale = np.exp(np.log(prob) / temperature)
            for j in range(batch_size):
                p = rescale[j, :]
                p[:] /= p.sum()
                outputs_batch[i][j] = _choice(tmp, p)
                # outputs_batch[i][j] = np.random.choice(vocab, 1, p)
        else:
            outputs_batch[i][:] = np.argmax(prob, axis=1)
        X_input_batch[:] = outputs_batch[i]
    return outputs_batch


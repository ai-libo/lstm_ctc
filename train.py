# encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import common
import utils
import pdb
from utils import decode_sparse_tensor
from tensorflow.python import debug as tfdbg

num_classes = len(common.CHARSET) + 1 #781
num_epochs = 2000
num_hidden = 200
num_layers = 1
train_inputs, train_targets, train_seq_len = utils.get_data_set('train')
val_inputs, val_targets, val_seq_len = utils.get_data_set('val')
print("Data loaded....")

# graph = tf.Graph()
def report_accuracy(decoded_list, train_targets):
    original_list = decode_sparse_tensor(train_targets)#list,length:100,['6', '0', '6', '/','2', '血', '1', '9', ' ', '1', '2', ':', '2', '2', '~']
    detected_list = decode_sparse_tensor(decoded_list)#list,lenth:100,预测序列['0', '2', '0', '2', '0', '2', '0', '2', '2', '2', '0', '2', '0', '2', '0', '2', '0', '2', '2', '2', '2', '2', '0', '2', '2', '0', '2']
    true_numer = 0
    # print(detected_list)
    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
    print("Test Accuracy:", true_numer * 1.0 / len(original_list))


def train():

    #第一部分：构建model
    global_step = tf.Variable(0, trainable=False)#全局步骤计数
    learning_rate = tf.train.exponential_decay(common.INITIAL_LEARNING_RATE,
                                               global_step,
                                               common.DECAY_STEPS,
                                               common.LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)

    inputs = tf.placeholder(tf.float32, [None, None, common.OUTPUT_SHAPE[0]]) #[,,60]
    targets = tf.sparse_placeholder(tf.int32)#三元组稀疏张量
    seq_len = tf.placeholder(tf.int32, [None])#list 64  [180,180,...,180]


    W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1), name="W")#shape (200,781)
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")#781

    #cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    #outputs1, _ = tf.nn.dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32)#(64, 3000, 200)
    cell_fw = tf.contrib.rnn.LSTMCell(num_hidden)
    cell_bw = tf.contrib.rnn.LSTMCell(num_hidden)
    initial_state_fw = cell_fw.zero_state(common.BATCH_SIZE,dtype=tf.float32)
    initial_state_bw = cell_bw.zero_state(common.BATCH_SIZE,dtype=tf.float32)
    (out, states)=tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,seq_len, initial_state_fw,initial_state_bw)
    outputs1 = tf.concat(out, 2)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1] #64,3000
    outputs = tf.reshape(outputs1, [-1, num_hidden])#(19200,200)
    logits0 = tf.matmul(outputs, W) + b
    logits1 = tf.reshape(logits0, [batch_s, -1, num_classes])
    logits = tf.transpose(logits1, (1, 0, 2))#(3000, 64, 781)

    loss = tf.nn.ctc_loss( targets, logits, seq_len)
    #inputs/logits: 3-D float Tensor.If time_major == True (default),  will shaped: [max_time x batch_size x num_classes]
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=common.MOMENTUM).minimize(cost, global_step=global_step)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)#or "tf.nn.ctc_greedy_decoder"一种解码策略
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))


    #第二部分：在session 中执行图
    with tf.Session() as sess:
        #封装　成可以调试的sess
        # sess = tfdbg.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
        for curr_epoch in range(num_epochs):
            print("Epoch.......", curr_epoch)
            train_cost = train_ler = 0
            for batch in range(common.BATCHES):
                start = time.time()
                train_inputs, train_targets, train_seq_len = utils.get_data_set('train', batch * common.BATCH_SIZE,
                                                                                  (batch + 1) * common.BATCH_SIZE)
                '''
                train_inputs:shape(64, 3000, 60)
                train_targets:3-tuple,sparse_tensor,(indices_matrix,values,shape)
                train_seq_len:length  64  [180,180,...,180]
                (Pdb) p train_targets
                    (array([[ 0,  0],
                            [ 0,  1],
                            [ 0,  2],
                            ..., 
                            [63, 0],
                            [63, 1],
                            [63, ?]]), array([25, 19, 19, ..., 19, 22,  2], dtype=int32), array([ 64, 145]))

                '''
                print("get data time", time.time() - start)
                start = time.time()
                train_feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
                b_cost, steps, _ = sess.run([cost, global_step, optimizer], train_feed)#训练
                #outputs1 = sess.run([outputs1],train_feed)
                '''
                outputs1:list,length:1 ouputs[0].shape:(64, 3000, 200)
                '''
                #outputs = sess.run([outputs],train_feed)
                '''
                outputs:list,length:1 ouputs[0].shape:(192000, 200)
                '''
                #logits = sess.run([logits],train_feed)
                '''
                logits:list,lenth=1,logits[0],shape:(3000, 64, 781)
                '''

                if steps > 0 and steps % common.REPORT_STEPS == 0:
                    #pdb.set_trace()
                    val_feed = {inputs: val_inputs,targets: val_targets,seq_len: val_seq_len}#64个验证样本
                    '''
                    val_inputs:shape(100, 3000, 60)
                    val_targets:3-tuple,sparse_tensor,(indices_matrix,values,shape)
                    val_seq_len:length  100  [180,180,...,180]
                    (Pdb) p train_targets
                        (array([[ 0,  0],
                                [ 0,  1],
                                [ 0,  2],
                                ..., 
                                [99, 0],
                                [99, 1],
                                [99, ?]]), array([25, 19, 19, ..., 19, 22,  2], dtype=int32), array([ 64, 145]))

                    '''
                    decoded0, log_probs, accuracy = sess.run([decoded[0], log_prob, acc], val_feed)
                    '''
                    decoded0:3-tuple
                    SparseTensorValue(indices=array([[ 0,  0],
                       [ 0,  1],
                       [ 0,  2],
                       ..., 
                       [99, 16]]), values=array([21, 20, 21, ..., 21, 21, 21]), dense_shape=array([100,  17]))
                    log_probs:shape (100, 1)
                    accuracy:0.92347372

                    '''
                    report_accuracy(decoded0, val_targets)
                save_path = saver.save(sess, "models/ocr.model", global_step=steps)#保存模型
               
                train_cost += b_cost * common.BATCH_SIZE
                seconds = time.time() - start
                print("Step:", steps, ", batch seconds:", seconds)

            train_cost /= common.TRAIN_SIZE

            val_feed = {inputs: val_inputs,
                        targets: val_targets,
                        seq_len: val_seq_len}

            val_cost, val_ler, lr, steps = sess.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)
            log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
            print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, train_ler, val_cost, val_ler,
                             time.time() - start, lr))


if __name__ == '__main__':
    train()

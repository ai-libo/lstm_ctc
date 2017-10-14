#CNN
x = tf.placeholder(tf.float32,[None,input_node],name="x_input")
y_ = tf.placeholder(tf.float32,[None,output_node],name="y_output")

#input-->layer1
w_1 = tf.Variable(tf.truncted_normal([input_node,L1_node],stdev=0.5))
b_1 = tf.Variable(tf.constant(0.1,shape=[L1_node]))
l_conv1 = tf.nn.relu(tf.matmul(x,w_1)+b_1,strides=[1,2,2,1])
l_pool1 = tf.nn.max_pool(l_conv1,strides=[1,2,2,1],ksize = [1,2,2,1],padding='SAME')

#layer1-->layder2
w_2 = tf.Variable(tf.truncted_normal([L1_node,L2_node],stddev=0.5))
b_2 = tf.Variable(tf.constant(0.1,shape=[L2_node]))
l_conv2 = tf.nn.relu(tf.matmul(l_pool1,w_2)+b_2)
l_pool2 = tf.nn.max_pool(l_conv2,strides=[1,2,2,1],ksize = [1,2,2,1],padding='SAME')

#layser2-->fc
w_3 = tf.Variable(tf.truncted_normal([L2_node,fc_node],stddev=0.5))
b_3 = tf.Variable(tf.constant(0.1,shape=[fc_node]))
l_3 = tf.reshape(l_pool2,[-1,])
fc_1 = tf.nn.relu(tf.matmul(l_3,w_3)+b_3)

#fc-->dropout
drop = tf.nn.dropout(fc_1,keep_prob)

#dropout-->softmax
w_4 = tf.Variable(tf.truncted_normal([fc_node,output_node],stddev=0.5))
b_4 = tf.Variable(tf.constant(0.1,shape=[output_node]))
y = tf.nn.softmax(tf.matmul(drop,w_4)+b_4)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.argmax(y,1),labels=tf.argmax(y_,1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy+reularation

train_step = tf.train.GradientDescentOptimizer(leraning_rate).minimize(loss)#以何种方式何种学习率去优化何种目标

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictiontf.float32))

with tf.session() as sess:
	tf.global_variable_initializer().run()

	for i in range(max_steps):
		sess.run(train_step,feed_dict={x:,y:})

		if i%1000 == 0:
			validate_accu = sess.run(accuracy,feed_dict={x:x_val,y:y_val})

	test_accu = sess.run(accuracy,feed_dict = {x:x_test,y:y_dict})

#RNN
input_size = 28(28*28 image)
hidden_size = 256
layer_num = 2
class_num =10

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,class_num])
keep_prob = tf.placeholder(tf.float32)

x = tf.reshape(x,[-1,28,28])

#一层lstm
lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,forget_bias=1.0,..)

#添加dropout
lstm_layer = tf.contrib.rnn.DropoutWrrapper(lstm_layer,input_keep_prob =1.0,output_keep_prob=keep_prob)

#堆叠多层
mlstm = tf.contrib.rnn.MultiRNNCell([lstm_layer]*layer_sum,...)

init_state = mlstm.zero_state(batch_size,dtype=tf.float32)

output = mlstm(x)

#添加softmax层
w = tf.Variable(tf.truncted_normal([hidden_size,class_num],stddev=0.1),dtype=tf.float32)
b = tf.Variable(tf.constant(0.1,shape=[class_num]),dtype=tf.float32)
y_ = tf.nn.softmax(tf.matmul(output,w)+b)

cross_entropy = tf.reduce_mean(-y*tf.log(y_))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#tf.argmax(input, axis=None, name=None, dimension=None)此函数是对矩阵按行或列计算最大值,0:按列，此处按行
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))#tf.cast():数据格式转换，此处bool-->float

with tf.Session as sess:

	sess.run(train_step,feed_dict={x:,y:,keep_prob:}) #train

	if i%1000 ==0:
		train_accuracy = sess.run(accuracy,feed_dict={x:x_val,y:y_val,keep_prob:})
		print(train_accuracy)

	#测试集	
	test_accuracy = sess.run(accuracy,feed_dict={x:x_test,y:y_test,keep_prob:})
	print(test_accuracy)


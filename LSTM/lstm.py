from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn

#导入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_DATA/',one_hot=True)

#设置超参数
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

#设置网络参数
n_input = 28
n_steps = 28 #这样就相当于遍历了整张图片，并且可认为带有时间的信息
n_hidden = 128 #特征的隐层数
n_classes = 10

#设置占位符
X = tf.placeholder("float",[None,n_steps,n_input])
Y = tf.placeholder("float",[None,n_classes])

#定义权重
weights = {
	'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))
}
biases = {
	'out':tf.Variable(tf.random_normal([n_classes]))
}

#定义网络
def RNN(X,weights,biases):
	#需要确认一下数据尺寸来匹配RNN函数的需求
	#当前输入的尺寸[batch_size,n_steps,n_input]
	#需要的尺寸：[batch_size,n_input]
	X = tf.unstack(X,n_steps,1)
	
	#定义一个LSTM细胞
	lstm_cell = rnn.BasicLSTMCell(n_hidden,forget_bias = 1.0)
	
	#获取LSTM细胞的输出
	outputs,states = rnn.static_rnn(lstm_cell,X,dtype = tf.float32)
	
	#线性激活，输出
	return tf.matmul(outputs[-1],weights['out'])+biases['out']

prediction = RNN(X,weights,biases)

#定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=Y))

#优化函数
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#评估模型
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#初始化变量
init = tf.global_variables_initializer()

#启动图表
with tf.Session() as sess:
	sess.run(init)
	step = 1
	#开始迭代
	while step * batch_size < training_iters:
		batch_X,batch_Y = mnist.train.next_batch(batch_size)
		#改变每个batch的尺寸
		batch_X = batch_X.reshape((batch_size,n_steps,n_input))
		#优化
		sess.run(optimizer,feed_dict={X:batch_X,Y:batch_Y})
		
		#计算每个batch的损失
		if step%display_step == 0:
			loss,acc = sess.run([cost,accuracy],feed_dict={X:batch_X,Y:batch_Y})
			print("Iter "+str(step*batch_size)+", Minibatch loss="+"{:.6f}".format(loss)
			+",Training accuracy="+"{:.5f}".format(acc))
		
		step += 1
	print("Optimization Finished!\n")
	
	#计算测试数据集上的精度
	test_len = 150
	test_data = mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
	test_label = mnist.test.labels[:test_len]
	print("Testing Accuracy:",sess.run(accuracy,feed_dict={X:test_data,
														   Y:test_label}))
		

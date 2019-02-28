from __future__ import print_function
import tensorflow as tf

#获取mnist数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_DATA/',one_hot=True)


#超参数
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

#网络参数
p_input = 784
n_classes = 10
dropout = 0.75 #保留0.75的神经元

#占位符
X = tf.placeholder(tf.float32,[None,p_input])
Y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)

#建立卷积函数和池化函数
def conv2d(X,W,b,strides=1):
	#卷积之后加上bias和relu激活函数
	X = tf.nn.conv2d(X,W,strides=[1,strides,strides,1],padding='SAME')
	X = tf.nn.bias_add(X,b)
	return tf.nn.relu(X)
	
def maxpool2d(X,k=2):
	return tf.nn.max_pool(X,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

#创建模型两个卷积层，两个池化层和一个全连接层
def conv_net(X,weights,biases,dropout):
	#重新定义输入图片的形状
	X = tf.reshape(X,shape=[-1,28,28,1])
	
	#卷积层1
	conv1 = conv2d(X,weights['wc1'],biases['bc1'])
	#最大池化层1
	pool1 = maxpool2d(conv1,k=2)

	#卷积层2
	conv2 = conv2d(pool1,weights['wc2'],biases['bc2'])
	#最大池化层2
	pool2 = maxpool2d(conv2,k=2)
	
	#全连接层
	fc1 = tf.reshape(pool2,[-1,weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	#使用dropout
	fc1 = tf.nn.dropout(fc1,dropout)
	
	#输出
	out = tf.add(tf.matmul(fc1,weights['out']),biases['out'])
	return out
	
#权重和偏差
weights = {
	#卷积核大小：5*5，输入大小：1，输出大小：32
	'wc1':tf.Variable(tf.random_normal([5,5,1,32])),
	#卷积核大小：5*5，输入大小：32，输出大小：64
	'wc2':tf.Variable(tf.random_normal([5,5,32,64])),
	#全连接层，输入大小：7*7*64，输出大小：1024
	'wd1':tf.Variable(tf.random_normal([7*7*64,1024])),
	#输出层，输入大小：1024，输出大小：10
	'out':tf.Variable(tf.random_normal([1024,n_classes]))
}

biases = {
	'bc1':tf.Variable(tf.random_normal([32])),
	'bc2':tf.Variable(tf.random_normal([64])),
	'bd1':tf.Variable(tf.random_normal([1024])),
	'out':tf.Variable(tf.random_normal([n_classes]))
}

#构建模型
prediction = conv_net(X,weights,biases,keep_prob)

#损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=Y))

#优化函数
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#模型评估
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#初始化变量
init = tf.global_variables_initializer()

#启动图表
with tf.Session() as sess:
	sess.run(init)
	step = 1
	
	#训练
	while step*batch_size < training_iters:
		batch_X,batch_Y = mnist.train.next_batch(batch_size)
		#进行优化
		sess.run(optimizer,feed_dict={X:batch_X,Y:batch_Y,keep_prob:dropout})
		
		if step%display_step == 0:
			loss,acc = sess.run([cost,accuracy],feed_dict={X:batch_X,Y:batch_Y,keep_prob:1.})
			print("Iter "+str(step*batch_size)+", Minibatch loss="+"{:.6f}".format(loss)
			+",Accuracy="+"{:.5f}".format(acc))
		
		step += 1
	print("Optimization Finished!\n")
	
	print("Testing Accuracy:",sess.run(accuracy,feed_dict={X:mnist.test.images[:256],
														   Y:mnist.test.labels[:256],
														   keep_prob:1.}))
			

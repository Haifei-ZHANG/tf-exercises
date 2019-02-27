from __future__ import print_function

#获取mnist数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_DATA/',one_hot=True)

import tensorflow as tf

#设置参数
learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 1

#设置网络参数
n_hidden_1 = 256 #第一个隐藏层的结点个数
n_hidden_2 = 256 #第二个隐藏层的结点个数
p_input = 784 #输入的数据是28*28共784维的
n_classes = 10 #一共有0-9是个类别

#设置占位符
X = tf.placeholder("float",[None,p_input])
Y = tf.placeholder("float",[None,n_classes])

#创建网络
def multilayer_perceptron(X, weights, biases):
	#前向传播，使用relu激活函数
	layer_1 = tf.add(tf.matmul(X,weights['h1']),biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	
	layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	
	#输出层使用线性激活函数
	out_layer = tf.matmul(layer_2,weights['out'])+biases['out']
	return out_layer

#初始化参数weights和biases
weights = {
	'h1':tf.Variable(tf.random_normal([p_input,n_hidden_1])),
	'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
	'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}
biases = {
	'b1':tf.Variable(tf.random_normal([n_hidden_1])),
	'b2':tf.Variable(tf.random_normal([n_hidden_2])),
	'out':tf.Variable(tf.random_normal([n_classes]))
}

#构建模型
prediction = multilayer_perceptron(X, weights, biases)

#定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=Y))

#定义优化方法
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#初始化变量
init = tf.global_variables_initializer()

#定义一个session
with tf.Session() as sess:
	sess.run(init)
	
	#循环训练
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		
		for i in range(total_batch):
			batch_X, batch_Y = mnist.train.next_batch(batch_size)
			
			#提供数据并运行optimizer和cost
			_, c = sess.run([optimizer,cost],feed_dict={X:batch_X,Y:batch_Y})
			
			#计算平均损失函数
			avg_cost += c/total_batch
			
		#每个epoch step显示log
		if epoch%display_step == 0:
			print("Epoch:",'%04d'%(epoch+1),"cost=","%0.5f"%avg_cost)
	print("\nOptimization Finished!\n")
	
	#测试模型
	correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
	#计算精度
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
	print("\nAccuracy:",accuracy.eval({X:mnist.test.images,Y:mnist.test.labels}))
	out_result = sess.run(prediction,feed_dict={X:mnist.test.images})
	print(sess.run(tf.argmax(out_result,1)))

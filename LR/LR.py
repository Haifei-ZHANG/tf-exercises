#用TensorFlow实现普通线性回归

from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("import succes")

#这个模型的超参数只有学习率，需要手动设置
learning_rate = 0.01
#设置训练的最大迭代次数
iteration_number = 1000
#设置显示间隔
display_step = 50

#模拟生成训练数据，X是简单的一维数据
train_X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,
1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
#样本数量
n_train = train_X.shape[0]

#创建placeholder
X = tf.placeholder("float")
Y = tf.placeholder("float")

#创建模型的参数，在tf中用变量表示
W = tf.Variable(np.random.randn(),name="weight")
b = tf.Variable(np.random.randn(),name="bias")

#创建决策函数
prediction = tf.add(tf.multiply(X,W),b)

#创建损失函数
loss = tf.reduce_sum(tf.pow(prediction-Y,2))/(2*n_train)

#创建优化方法
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#模型参数初始化
init = tf.initialize_all_variables()
#init = global_variables_initializer()

#创建计算图
with tf.Session() as sess:
	sess.run(init)
	
	#创建迭代循环
	for iteration in range(iteration_number):
		for (x,y) in zip(train_X,train_Y):
			#通过feed_dict将数据喂给模型
			sess.run(optimizer, feed_dict={X:x, Y:y})
			
			#每隔50次显示一下损失函数的值
			if (iteration+1)%display_step == 0:
				c = sess.run(loss,feed_dict={X:train_X, Y:train_Y})
				print("iteration:",'%4d'%(iteration+1),"loss=","{:.9f}".
				format(c),"\n","W=",sess.run(W),"b=",sess.run(b),"\n")
	
	print("Optimization finished!")
	#训练结束，打印一下最终的损失函数的值
	training_loss = sess.run(loss,feed_dict={X:train_X, Y:train_Y})
	print("Training loss=",training_loss,"W=",sess.run(W),"b=",sess.run(b))
	
	#用matplot生成一幅图
	plt.plot(train_X,train_Y,'ro',label='Original data')
	plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted line')
	plt.legend()
	plt.show()
	
	#生成一段测试数据
	test_X = np.array([6.83,4.668,8.9,7.91,5.7,8.7,3.1,2.1])
	test_Y = np.array([1.84,2.273,3.2,2.831,2.92,3.24,1.35,1.03])
	
	print("Testing....(Mean square loss Comparison)")
	#计算在测试数据集上的损失值
	test_loss = sess.run(tf.reduce_sum(tf.pow(prediction-Y,2))/(2*test_X
	.shape[0]),feed_dict={X:test_X,Y:test_Y})
	print("Test loss=",test_loss)
	print("Absolute mean square loss difference:",abs(training_loss-test_loss))
	plt.plot(test_X,test_Y,'bo',label="Testing data")
	plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted line')
	plt.legend()
	plt.show()

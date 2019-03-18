from __future__ import print_function

#获取mnist数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_DATA/',one_hot=True)

import tensorflow as tf

#模型超参数
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1
logs_path = '/tmp/tensorflow_logs/example_advanced'

#网络参数
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

#创建占位符
X= tf.placeholder("float",[None,n_input],name='InputData')
Y= tf.placeholder("float",[None,n_classes],name='LabelData')

print("Begin")
#创建计算图
#创建第一个隐层
with tf.name_scope('ReluLayer1'):
	weights_h1 = tf.Variable(tf.random_normal([n_input,n_hidden_1]),name='weights_h1')
	biases_b1 = tf.Variable(tf.random_normal([n_hidden_1]),name='biases_b1')
	Matmul_1 = tf.matmul(X,weights_h1)
	BiasAdd_1 = tf.add(Matmul_1,biases_b1)
	Relu_1 = tf.nn.relu(BiasAdd_1)
 
#创建第二个隐层
with tf.name_scope('ReluLayer2'):
	weights_h2 = tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2]),name='weights_h2')
	biases_b2 = tf.Variable(tf.random_normal([n_hidden_2]),name='biases_b2')
	Matmul_2 = tf.matmul(Relu_1,weights_h2)
	BiasAdd_2 = tf.add(Matmul_2,biases_b2)
	Relu_2 = tf.nn.relu(BiasAdd_2)
	
#创建Logistic层
with tf.name_scope('LogitLayer'):
	weights_out = tf.Variable(tf.random_normal([n_hidden_2,n_classes]),name='weights_out')
	biases_out = tf.Variable(tf.random_normal([n_classes]),name='biases_out')
	Matmul_3 = tf.matmul(Relu_2,weights_out)
	BiasAdd_3 = tf.add(Matmul_3,biases_out)

#创建Softmax分类预测层
with tf.name_scope('softmax'):
	pred = tf.nn.softmax(BiasAdd_3)

#定义损失函数
with tf.name_scope('CrossEntropy'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=BiasAdd_3,labels=Y))

#定义优化方法
with tf.name_scope('SGD'):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	#定义每个梯度操作
	grads = tf.gradients(cost,tf.trainable_variables())
	grads = list(zip(grads,tf.trainable_variables()))
	#根据梯度修改参数
	apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

with tf.name_scope('Accuracy'):
	acc = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
	acc = tf.reduce_mean(tf.cast(acc,tf.float32))
	
#初始化参数
init = tf.global_variables_initializer()

#可视化交叉熵损失函数
tf.summary.scalar('Cross Entropy', cost)
#可视化模型精度
tf.summary.scalar('Accuracy', acc)

#可视化模型参数
for var in tf.trainable_variables():
	tf.summary.histogram(var.name,var)

#可视化梯度
for grad,var in grads:
	tf.summary.histogram(var.name+'/gradient/',grad)

merged_summary_op = tf.summary.merge_all()

#运行计算图
with tf.Session() as sess:
	sess.run(init)
	
	#将日志写入TensorBoard
	summary_writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())
	
	#创建循环
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		
		for i in range(total_batch):
			batch_X,batch_Y = mnist.train.next_batch(batch_size)
			#进行反馈计算
			_, c, summary = sess.run([apply_grads, cost, merged_summary_op],feed_dict={X:batch_X, Y:batch_Y})
			summary_writer.add_summary(summary, epoch*total_batch+i)
			#计算平均损失
			avg_cost += c/total_batch
		#显示训练过程
		if epoch%display_step == 0:
			print("Epoch:",'%04d'%(epoch+1),"cost=","{:.6f}".format(avg_cost))
	print("Optimization Finished!\n")

	#测试模型性能
	correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
	#计算精度
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	print("Accuracy:",accuracy.eval({X:mnist.test.images,Y:mnist.test.labels}))
	
	print("Run the command line:\n"\
	      "-->tensorboard --logdir=/tmp/tensorflow_logs/example_advanced"\
	      "\nThen open http://0.0.0.0:6006/ into your web browser")

		

	

print('End')

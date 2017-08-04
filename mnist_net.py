import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Define the Network - The Computation Graph
input_layer = tf.placeholder(tf.float32, [None, 784])
input_weight = tf.Variable(tf.zeros([784, 10]))
input_biases = tf.Variable(tf.zeros([10]))
output = tf.matmul(input_layer, input_weight) + input_biases
labels = tf.placeholder(tf.float32, [None, 10])

#Train the Network
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))
train = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

#We need the session so run the Computation Graph above
session = tf.InteractiveSession()

#Initialize all Variables
tf.global_variables_initializer().run()

#This is where we feed the data into the network
for _ in range(1000):
    batch_instances, batch_labels = mnist.train.next_batch(100)
    session.run(train, feed_dict={input_layer: batch_instances, labels: batch_labels})

#Testing the Network
nets_answer = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(nets_answer, tf.float32))

print(session.run(accuracy, feed_dict={input_layer: mnist.test.images, labels: mnist.test.labels}))

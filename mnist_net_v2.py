import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Defining the Network
input_layer = tf.placeholder(tf.float32, [None, 784])
input_layer_weights = tf.Variable(tf.zeros([784, 10]))
input_layer_biases = tf.Variable(tf.zeros([10]))

output = tf.matmul(input_layer, input_layer_weights) + input_layer_biases

labels = tf.placeholder(tf.float32, [None, 10])

#Training the Network
#Define cost function, then train

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = output))
train_algo = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
    batch_instance, batch_label = mnist_data.train.next_batch(100)
    sess.run(train_algo, feed_dict={input_layer: batch_instance, labels: batch_label})

print("Training Complete. Starting Testing")

network_answer = sess.run(tf.argmax(output, 1), feed_dict={input_layer: mnist_data.test.images, labels:mnist_data.test.labels})
test_labels = sess.run(tf.argmax(mnist_data.test.labels, 1), feed_dict={input_layer: mnist_data.test.images, labels:mnist_data.test.labels})

matrix = confusion_matrix(test_labels, network_answer)
precision = precision_score(test_labels, network_answer, average=None)
recall = recall_score(test_labels, network_answer, average=None)
f1 = f1_score(test_labels, network_answer, average=None)

print("The Matrix:")
print(matrix)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

#print("Output: ", sess.run(network_answer, feed_dict={input_layer: mnist_data.test.images, labels:mnist_data.test.labels}))
#print("Labels: ", sess.run(test_labels, feed_dict={input_layer: mnist_data.test.images, labels:mnist_data.test.labels}))

print("Program Complete")

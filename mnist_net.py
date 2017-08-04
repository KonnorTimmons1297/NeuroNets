import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Network():
  
  def __init__(self):
    self.mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    
    self.input_layer = tf.placeholder(tf.float32, [None, 784])
    self.input_weights = tf.Variable(tf.zeros([784, 10]))
    self.input_biases = tf.Variable(tf.zeros([10]))
    
    self.output = tf.nn.softmax(tf.matmul(self.input_layer, self.input_weights) + self.input_biases)
    
  def train(self):
    this.session = tf.Session()
    
    tf.global_variables_initializer().run()
    
    for _ in range(1000):
      batch_instances, batch_labels = mnist.train.next_batch(100)
      session.run(self.train, feed_dict={self.input_layer: batch_instances, output: batch_labels})
  
  def train_algo(self):
    self.labels = tf.placeholder(tf.float32, [None, 10])
    
    self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.output))
    
    self.train = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
    
  def test(self):
    nets_answer = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(nets_answer, tf.float32))
    
    print(session.run(accuracy, feed_dict={input_layer: mnist.test.images, labels: mnist.test.labels})
          
          
          
          
net = Network()
net.train()
net.test()

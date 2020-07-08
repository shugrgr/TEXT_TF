import tensorflow as tf
import shutil
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from linear import Linear,Linear_config

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/mnist', one_hot=True)

X_test = mnist.test.images
Y_test = mnist.test.labels


config = Linear_config()
save_path = os.path.join('./data/checkpoint', config.model_name)
model = Linear(config)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
model_file = tf.train.latest_checkpoint(save_path)
saver.restore(sess=sess, save_path=model_file)

pred, acc = sess.run([model.pred, model.acc], feed_dict={model.X:X_test, model.Y:Y_test})
print(acc)
print(pred)
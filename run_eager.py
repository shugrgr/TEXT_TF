import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from linear import Linear,Linear_config
tf.enable_eager_execution()
tfe = tf.contrib.eager
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/mnist', one_hot=True)

X_test = mnist.test.images
Y_test = mnist.test.labels

config = Linear_config()
model = Linear(config)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print('start')
for epoch in range(config.epoch):
    batch_X, batch_Y = mnist.train.next_batch(config.batch_size)
    sess.run([model.optim], feed_dict={model.X:batch_X, model.Y:batch_Y})

    if epoch % config.display_setp == 0:
        loss, acc = sess.run([model.loss, model.acc], feed_dict={model.X:batch_X, model.Y:batch_Y})
        test_loss, test_acc = sess.run([model.loss, model.acc], feed_dict={model.X:X_test, model.Y:Y_test})
        print("epoch {0:6} training loss {1:.4f}, acc {2:}.".format(epoch, loss, acc))
        print("epoch {0:6} test loss {1:6}, acc {2:6}.".format(epoch, test_loss, test_acc))



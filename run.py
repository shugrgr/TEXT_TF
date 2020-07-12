import tensorflow as tf
import shutil
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from linear import Linear,Linear_config
from cnn import CNN,CNN_config

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/mnist', one_hot=True)

X_test = mnist.test.images
Y_test = mnist.test.labels


# config = Linear_config()
# model = Linear(config)

config = CNN_config()
model = CNN(config)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print('start')

tensorboard_dir = os.path.join('./data/tensorboard', config.model_name)
if os.path.exists(tensorboard_dir):
    shutil.rmtree(tensorboard_dir)
os.makedirs(tensorboard_dir)
tensorboard_train_dir = os.path.join(tensorboard_dir, config.model_name+'_train')
tensorboard_test_dir = os.path.join(tensorboard_dir, config.model_name+'_test')
os.makedirs(tensorboard_train_dir)
os.makedirs(tensorboard_test_dir)
tf.summary.scalar("loss", model.loss)
tf.summary.scalar("accuracy", model.acc)
train_writer = tf.summary.FileWriter(tensorboard_train_dir, sess.graph)
test_writer = tf.summary.FileWriter(tensorboard_test_dir, sess.graph)
merged_summary = tf.summary.merge_all()

# 配置 Saver
saver = tf.train.Saver(max_to_keep=2)
save_path = os.path.join('./data/checkpoint', config.model_name)
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
save_path = os.path.join(save_path, config.model_name)

best_acc_val = 0.0  # 最佳验证集准确率
last_improved = 0  # 记录上一次提升批次
require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练
break_epoch = 2000

for epoch in range(config.epoch):
    batch_X, batch_Y = mnist.train.next_batch(config.batch_size)
    sess.run([model.optim], feed_dict={model.X:batch_X, model.Y:batch_Y, model.dropout:config.dropout})

    if epoch % 100 == 0:
        loss, acc, s= sess.run([model.loss, model.acc, merged_summary], feed_dict={model.X:batch_X, model.Y:batch_Y, model.dropout:0})
        train_writer.add_summary(s, epoch)

        print("epoch {0:>6} training loss {1:>6.2}, acc {2:>7.2%}.".format(epoch, loss, acc))
    if (epoch+1) % config.display_setp == 0:
        test_loss, test_acc, s = sess.run([model.loss, model.acc, merged_summary], feed_dict={model.X:X_test, model.Y:Y_test, model.dropout:0})
        test_writer.add_summary(s, epoch)
        if test_acc > best_acc_val:
            best_acc_val = test_acc
            saver.save(sess=sess, save_path = save_path, global_step=epoch+1)
            last_improved = epoch
            improve_string = '*'
        else:
            improve_string = ''
        print('*'*50)
        print("epoch {0:>6} test loss {1:>6.2}, acc {2:>7.2%} {3}.".format(epoch, test_loss, test_acc, improve_string))
        print('*'*50)
    if epoch - last_improved >= break_epoch:
        break



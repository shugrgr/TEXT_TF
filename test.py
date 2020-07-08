import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow.contrib.eager as tfe

# a=tf.constant(2)
# b=tf.constant(3)

# with tf.Session() as sess:
#     print(sess.run(a))
#     print(sess.run(b))
#     print('*'*50)

# a=tf.placeholder(tf.int16)
# b=tf.placeholder(tf.int16)

# add = tf.add(a,b)
# mul = tf.multiply(a,b)
# with tf.Session() as sess:
#     print(sess.run(add,feed_dict={a:2, b:3}))
#     print(sess.run(mul,feed_dict={a:2, b:3}))
#     print('*'*50)

# a=tf.constant([[3,3]])
# b=tf.constant([[2],[2]])
# c=tf.matmul(a,b)
# with tf.Session() as sess:
#     print(sess.run(c))

print("Setting Eager mode...")
tfe.enable_eager_execution()

a=tf.constant(2)
b=tf.constant(3)
print(a)
c=a+b
print(c)


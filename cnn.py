import tensorflow as tf

class CNN_config(object):
    def __init__(self):
        self.model_name = 'CNN'
        self.hidden_size = 128
        self.lr_rate = 1e-3
        self.batch_size = 128
        self.input_size = 28 * 28
        self.output_size = 10
        self.epoch = 10000
        self.display_setp = 500
        self.dropout = 0.5
        self.num_filters = 256  # 卷积核数目
        self.kernel_size = [2,3]  # 卷积核尺寸

        

class CNN(object):
    def __init__(self,config):
        self.config = config
        self._init_layer()
    
    def _init_layer(self):
        self.X = tf.placeholder(tf.float32, shape=[None ,self.config.input_size], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None ,self.config.output_size], name='Y')
        self.dropout = tf.placeholder(tf.float32)
        self.X_ = tf.reshape(self.X,[-1,28,28,1])

        with tf.name_scope("cnn"):
            # CNN layer
            self.conv2 = tf.layers.conv1d(self.X_, self.config.num_filters, 2, name='conv2',activation=tf.nn.relu, padding='same')
            self.conv3 = tf.layers.conv1d(self.X_, self.config.num_filters, 3, name='conv3',activation=tf.nn.relu, padding='same')
            # global max pooling layer
            self.gmp2 = tf.reduce_max(self.conv2, reduction_indices=[1], name='gmp2')
            self.gmp3 = tf.reduce_max(self.conv3, reduction_indices=[1], name='gmp3')
            self.cnn_output = tf.concat([self.gmp2,self.gmp3], 1)

        with tf.name_scope("layer1"):
            self.layer1 = tf.layers.dense(tf.nn.dropout(self.cnn_output, self.dropout), self.config.hidden_size, activation=tf.nn.relu, name='layer1')
        with tf.name_scope("layer2"):
            self.layer2 = tf.layers.dense(self.layer1, self.config.output_size, activation=None, name='layer2')
        self.pred = tf.argmax(tf.nn.softmax(self.layer2),1)
        with tf.name_scope("optim"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.layer2, labels=self.Y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.lr_rate).minimize(self.loss)
        with tf.name_scope("eval"):
            self.correct_pred = tf.equal(tf.argmax(self.Y, 1), self.pred)
            self.cast = tf.cast(self.correct_pred, tf.float32)
            self.acc = tf.reduce_mean(self.cast)

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
    config = CNN_config()
    model = CNN(config)

    sess=tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)


    # for tv in tf.trainable_variables():
    #     print (tv.name, tv.shape)

    batch_X, batch_Y = mnist.train.next_batch(1)
    drop = 0.5
    a,b,c,d,e=sess.run([model.conv2, model.conv3,model.gmp2, model.gmp3,model.cnn_output], feed_dict={model.X:batch_X, model.Y:batch_Y, model.dropout:drop})
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)
    print(e.shape)

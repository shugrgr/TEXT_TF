import tensorflow as tf

class CNN_config(object):
    def __init__(self):
        self.model_name = 'CNN'
        self.hidden_size = 128
        self.lr_rate = 1e-3
        self.batch_size = 64
        self.input_size = 28 * 28
        self.output_size = 10
        self.epoch = 10000
        self.display_setp = 500
        self.dropout = 0.5

        

class CNN(object):
    def __init__(self,config):
        self.config = config
        self._init_layer()
    
    def _init_layer(self):
        self.X = tf.placeholder(tf.float32, shape=[None ,self.config.input_size], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None ,self.config.output_size], name='Y')
        with tf.name_scope("layer1"):
            self.layer1 = tf.layers.dense(self.X, self.config.hidden_size, activation=tf.nn.relu, name='layer1')
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



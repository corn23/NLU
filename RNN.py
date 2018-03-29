import tensorflow as tf


class RNNmodel():
    def __init__(self,embedding_size, sequency_length, hidden_size, vocab_len):

        self.embedding_size = embedding_size
        self.sequence_length = sequency_length
        self.hidden_size = hidden_size
        self.vocab_len = vocab_len

        self.input_x = tf.placeholder(dtype=tf.int64, shape=[None, self.sequence_length])
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None, self.sequence_length])
        self.sequence_length_list = tf.placeholder(dtype=tf.int32, shape=[None,])
        self.sequence_mask = tf.sequence_mask(self.sequence_length_list,self.sequence_length,dtype=tf.float32)

        self.word_embeddings = tf.get_variable("word_embeddings", [self.vocab_len, self.embedding_size])
        self.embedded_tokens = tf.nn.embedding_lookup(self.word_embeddings, self.input_x)
        # split by the timestamp
        self.embedded_tokens = tf.unstack(self.embedded_tokens, num=self.sequence_length, axis=1)
        rnncell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.hidden_size)
        outputs, state = tf.nn.static_rnn(rnncell, self.embedded_tokens,dtype=tf.float32)

        self.outputs = tf.reshape(outputs,[-1,hidden_size])
        self.W_out = tf.get_variable("W_out", shape=[self.hidden_size, self.vocab_len],
                                initializer=tf.contrib.layers.xavier_initializer())
        self.b_out = tf.Variable(tf.constant(0.1, shape=[self.vocab_len,]), name='b_out')

        logits = tf.nn.xw_plus_b(self.outputs,self.W_out,self.b_out)
        logits = tf.reshape(logits, shape=[self.sequence_length, -1, self.vocab_len])
        self.logits = tf.transpose(logits, perm=[1, 0, 2])
        self.prediction = tf.argmax(logits, 1, name='prediction')
        self.loss = tf.contrib.seq2seq.sequence_loss(
                            self.logits,
                            self.input_y,
                            self.sequence_mask,
                            average_across_timesteps=True,
                            average_across_batch=True)


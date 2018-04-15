import tensorflow as tf


class RNNmodel():
    def __init__(self,embedding_size, sequency_length, hidden_size, vocab_len, batch_size, is_add_layer):

        self.embedding_size = embedding_size
        self.sequence_length = sequency_length
        self.hidden_size = hidden_size
        self.vocab_len = vocab_len
        self.batch_size = batch_size

        self.input_x = tf.placeholder(dtype=tf.int64, shape=[None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None, self.sequence_length], name="input_y")
        self.sequence_length_list = tf.placeholder(dtype=tf.int32, shape=[None,],name='sequence_length_list')
        self.sequence_mask = tf.sequence_mask(self.sequence_length_list, self.sequence_length,dtype=tf.float32)

        self.word_embeddings = tf.get_variable("word_embeddings", [self.vocab_len, self.embedding_size])
        self.embedded_tokens = tf.nn.embedding_lookup(self.word_embeddings, self.input_x)

        # split by the timestamp
        self.embedded_tokens = tf.unstack(self.embedded_tokens, num=self.sequence_length, axis=1)
        # rnncell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.hidden_size)
        with tf.variable_scope("rnn"):
            if is_add_layer:
                rnncell = tf.nn.rnn_cell.LSTMCell(num_units=2*self.hidden_size)
                W_middle = tf.get_variable("W_middle", shape=[2 * self.hidden_size, hidden_size],
                                           initializer=tf.contrib.layers.xavier_initializer())
            else:
                rnncell = tf.nn.rnn_cell.LSTMCell(num_units=2*self.hidden_size)
            state = rnncell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            outputs = []
            for _input in self.embedded_tokens:
                output,state = rnncell(_input,state)
                outputs.append(output)

        if is_add_layer:
            outputs = tf.reshape(outputs,[-1,2*hidden_size])
            self.outputs = tf.matmul(outputs,W_middle)
        else:
            self.outputs = tf.reshape(outputs,[-1,hidden_size])  # shape: (batch_size*time_step, hidden_size)
        self.W_out = tf.get_variable("W_out", shape=[self.hidden_size, self.vocab_len],
                                initializer=tf.contrib.layers.xavier_initializer())
        self.b_out = tf.Variable(tf.constant(0.1, shape=[self.vocab_len,]), name='b_out')

        logits = tf.nn.xw_plus_b(self.outputs,self.W_out, self.b_out)
        logits = tf.reshape(logits, shape=[self.sequence_length, -1, self.vocab_len])  # (time_step,batch_size,vocab_len)
        self.logits = tf.transpose(logits, perm=[1, 0, 2])
        self.prediction = tf.argmax(logits, 1, name='prediction')
        self.loss = tf.contrib.seq2seq.sequence_loss(
                            self.logits,
                            self.input_y,
                            self.sequence_mask,
                            average_across_timesteps=True,
                            average_across_batch=False,name="loss")

        self.eva_perplexity = tf.exp(self.loss, name="eva_perplexity")
        self.minimize_loss = tf.reduce_mean(self.loss,name="minize_loss")
        self.print_perplexity = tf.exp(self.minimize_loss,name="print_perplexity")

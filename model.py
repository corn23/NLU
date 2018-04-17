# Neural Language Model
import numpy as np
import tensorflow as tf
import gensim
import time
import os
import glob
import sys
from utils import id2word

def train(model,cfg,train_data=None,id2word_dict=None):

    learning_rate = cfg['learning_rate']
    is_use_embedding = cfg['is_use_embedding']
    embedding_path = cfg['embedding_path']
    vocab_len = cfg['vocab_len']
    max_grad_norm = cfg['max_grad_norm']
    embedding_dim = cfg['embedding_dim']
    num_epoch = cfg['num_epoch']

    # Training
    sess = tf.Session()
    with sess.as_default():
        # define the training process
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(model.minimize_loss, tvars), max_grad_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.apply_gradients(zip(grads, tvars),
            global_step=global_step)
        sess.run(tf.global_variables_initializer())

        #  add summary
        grad_summaries = []
        for g, v in zip(grads,tvars):
            if g is not None:
                grad_hist_summary = tf.summary.histogram('/grad/hist/%s' % v.name, g)
                grad_summaries.append(grad_hist_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # set the output dir
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'run', timestamp))
        print("write to {}\n".format(out_dir),flush=True)

        with open(os.path.join(out_dir,'log.txt'),'w') as f:
            for item in cfg.items():
                f.write(item)
                f.write('\n')
        # summary for the loss
        loss_summary = tf.summary.scalar("print_perplexity", model.print_perplexity)

        # train_summary
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        out_summary_dir = os.path.join(out_dir, "summary")
        train_summary_writer = tf.summary.FileWriter(out_summary_dir,sess.graph)

        # saver
        checkpoint_dir = os.path.join(out_dir,"checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # load embedding
        wordemb = gensim.models.KeyedVectors.load_word2vec_format(embedding_path,binary=False)
        my_embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_len,embedding_dim))
        if is_use_embedding:
            for id, word in id2word_dict.items():
                if word in wordemb.vocab:
                    my_embedding_matrix[id,:] = wordemb[word]
                else:
                    my_embedding_matrix[id,:] = np.random.uniform(-0.25, 0.25,embedding_dim)

        word_embedding = tf.placeholder(tf.float32,[None,None], name="pretrained_embeddings")
        set_x = model.word_embeddings.assign(word_embedding)

        sess.run(set_x, feed_dict={word_embedding:my_embedding_matrix})

        data_x,data_y,mask = train_data
        epoch = 0
        while epoch < num_epoch:
            batch_loss = 0
            for ibatch in range(data_x.shape[0]):
                _, summary, step, loss = sess.run([train_op, train_summary_op, global_step, model.print_perplexity],
                                                    feed_dict={model.input_x: data_x[ibatch,:,:],
                                                             model.input_y: data_y[ibatch,:,:],
                                                             model.sequence_length_list: mask[ibatch,:]})
                train_summary_writer.add_summary(summary=summary, global_step=step)
                print(ibatch, np.max(loss),flush=True)
                batch_loss += loss
                if ibatch%10 == 0:
                    valid_loss = sess.run([model.print_perplexity], feed_dict={model.input_x: data_x[-1, :, :],
                                                      model.input_y: data_y[-1, :, :],
                                                      model.sequence_length_list: mask[-1, :]})
                    print("ibatch", ibatch, "valid_loss",valid_loss,flush=True)
                    print("ibatch",ibatch,"train_loss",batch_loss/10,flush=True)
                    batch_loss=0
            sys.stdout.flush()
            epoch += 1
        saver.save(sess,checkpoint_dir,global_step=step)
    return out_dir

def evaluate(sess_path, eva_data, result_ptr):
    data_x,data_y,length_list = eva_data
    n_batch = len(data_x)
    sess = tf.Session()
    graph_path = os.path.join(sess_path,'*.meta')
    graph_name = glob.glob(graph_path)
    saver = tf.train.import_meta_graph(graph_name[0])
    saver.restore(sess, graph_name[0].split('.')[0])
    graph = tf.get_default_graph()

    # get ops
    perplexity = graph.get_tensor_by_name("eva_perplexity:0")
    input_x = graph.get_tensor_by_name("input_x:0")
    input_y = graph.get_tensor_by_name("input_y:0")
    sequence_length_list = graph.get_tensor_by_name("sequence_length_list:0")

    for ibatch in range(n_batch):
        this_perplexity = sess.run(perplexity, feed_dict={input_x: data_x[ibatch],
                                                          input_y: data_y[ibatch],
                                                          sequence_length_list: length_list[ibatch]})
        print(np.max(this_perplexity),np.argmax(this_perplexity),flush=True)
        for i_per in this_perplexity:
            result_ptr.write(str(i_per)+'\n')


def generate(sess_path, cfg, contin_data,result_ptr,id2word_dict):

    vocab_len = cfg['vocab_len']
    embedding_dim = cfg['embedding_dim']
    hidden_size = cfg['hidden_size']
    is_add_layer = cfg['is_add_layer']
    max_generate_length = cfg['max_generate_length']
    batch_size = cfg['batch_size']
    contin_data,_,length_list = contin_data
    contin_data = np.squeeze(contin_data, axis=1)  #we process sentence one by one
    length_list = np.squeeze(length_list, axis=1)
    sess = tf.Session()
    graph_path = os.path.join(sess_path,'*.meta')
    graph_name = glob.glob(graph_path)

    word_embeddings = tf.get_variable("word_embeddings", [vocab_len, embedding_dim])
    if is_add_layer:
        rnncell = tf.nn.rnn_cell.LSTMCell(num_units=2*hidden_size)
        W_middle = tf.get_variable("rnn/W_middle", shape=[2 * hidden_size, hidden_size])
    else:
        rnncell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)

    state=rnncell.zero_state(batch_size=1,dtype=tf.float32)
    with tf.variable_scope('rnn'):
        rnncell(tf.gather_nd(word_embeddings,[[0]]),state) # for create rnn kerner/ bias for parameter load

    W_out = tf.get_variable("W_out", shape=[hidden_size, vocab_len],
                                 initializer=tf.contrib.layers.xavier_initializer())
    b_out = tf.Variable(tf.constant(0.1, shape=[vocab_len, ]), name='b_out')

    saver = tf.train.Saver()
    ckpt_name = graph_name[0].split('.')[0]  # something like checkpoints-40
    saver.restore(sess,ckpt_name)
    sentence_id = 0
    for sentence_id_list,L in zip(contin_data,length_list):
        state = rnncell.zero_state(batch_size=1, dtype=tf.float32)

        for i in range(L):
            wordvec = tf.gather_nd(word_embeddings, [[sentence_id_list[i]]])
            output, state = rnncell(wordvec, state)

        generate_length = 0
        for i in range(L, max_generate_length):
            if is_add_layer:
                middle_output = tf.matmul(output,W_middle)
                final_output = tf.add(tf.matmul(middle_output, W_out),b_out)
                word_id = sess.run(tf.argmax(final_output,axis=1))
            else:
                word_id = sess.run(tf.argmax(tf.add(tf.matmul(output, W_out),b_out),axis=1))
            sentence_id_list[i]=word_id[0]
            generate_length += 1
            if word_id == 3: # <eos>
                break
            input = tf.nn.embedding_lookup(word_embeddings,word_id)
            output,state = rnncell(input,state)

        sentence = id2word(sentence_id_list,id2word_dict)
        print (sentence_id, ' '.join(sentence[:L+generate_length]),flush=True)
        result_ptr.write(' '.join(sentence[:L+generate_length])+'\n')
        sentence_id += 1

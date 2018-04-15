# Neural Language Model
import numpy as np
import tensorflow as tf
from RNN import RNNmodel
import gensim
import time
import os
import glob
import sys


vocab_len = 20000
num_epoch = 1
max_length = 30
batch_size = 50
embedding_dim = 100
hidden_size = 512
max_grad_norm = 5
text_num = 2000  # for quick dry-run
learning_rate = 0.01
is_add_layer = True
is_use_embedding = True


print("vocab_len:{} num_epch:{} text_num:{} learning_rate:{}".format(
vocab_len, num_epoch,text_num,learning_rate))
print("is_use_embedding",is_use_embedding)
print("is_add_layer",is_add_layer)

def load_data(path):
    f = open(path,'r')
    data = []
    for line in f:
        token = line.strip().split(' ')
        data.append(token)
    f.close()
    return data

def build_dict(data, vocab_len):
    # count the word frequency
    import collections
    word_counter = collections.Counter

    # to merger all token in one list
    all_word  = []
    for sentence in data:
        all_word.extend(sentence)

    pre_vocab = word_counter(all_word).most_common(vocab_len-4)
    # add <pad> <unk> <bos> <eos>
    pre_vocab_list = [word[0] for word in pre_vocab]
    vocab_list = ['<pad>', '<unk>', '<bos>', '<eos>']
    vocab_list.extend(pre_vocab_list)
    ID = range(vocab_len+1)
    word2id_dict = dict(zip(vocab_list, ID))
    id2word_dict = dict(zip(ID, vocab_list))

    return word2id_dict, id2word_dict


def word2id(sentence,word2id_dict):
    IDlist = []
    for word in sentence:
        if word in word2id_dict.keys():
            IDlist.append(word2id_dict[word])
        else:
            IDlist.append(1)  # <unk>

    return IDlist


def id2word(IDlist,id2word_dict):
    return [id2word_dict[id] for id in IDlist]

def add_special_string(IDlist, max_length):
    if len(IDlist) > max_length-2:
        IDlist = IDlist[:max_length-2]
    a = [2]  # <bos>
    a.extend(IDlist)
    a.append(3)  # <eos>
    while len(a) < max_length:
        a.append(0)
    return a

def convert_text_data(data, word2id_dict):
    # convert data into IDlist with the same shape
    IDdata = np.zeros(shape=(len(data),max_length), dtype=int)
    sequency_length = np.zeros(shape=(len(data),), dtype=int)
    target_data = np.zeros(shape=(len(data),max_length),dtype=int)
    for i, sentence in enumerate(data):
        sequency_length[i] = len(sentence)+1  # since we do not care what is after <eos>
        input_ID = add_special_string(word2id(sentence, word2id_dict), max_length=max_length)
        try:
            IDdata[i,:] = input_ID
        except ValueError:
            print(i)
        target_ID = input_ID[1:]
        target_ID.append(0)
        target_data[i,:] = target_ID
    return IDdata, target_data, sequency_length

def get_batch_data(data, batch_size):
    data_x, data_y, mask = data
    num_batch = len(data_x)//batch_size
    data_x = np.reshape(data_x[:num_batch*batch_size,:], [num_batch, batch_size, max_length])
    data_y = np.reshape(data_y[:num_batch*batch_size,:], [num_batch, batch_size, max_length])
    mask = np.reshape(mask[:num_batch*batch_size], [num_batch, batch_size])
    return data_x, data_y, mask

# evaluation phase
def evaluate(sess_path, data_x, data_y, length_list,result_ptr):
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
        for i_per in this_perplexity:
            result_ptr.write(str(i_per)+'\n')


def generate(sess_path, data_x, length_list, result_ptr,is_add_layer):
    data_x = np.squeeze(data_x, axis=1)  #we process sentence one by one
    length_list = np.squeeze(length_list, axis=1)
    sess = tf.Session()
    graph_path = os.path.join(sess_path,'*.meta')
    graph_name = glob.glob(graph_path)

    #word_embeddings = graph.get_tensor_by_name("input_x:0")
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

    for sentence_id_list,L in zip(data_x,length_list):
        state = rnncell.zero_state(batch_size=1, dtype=tf.float32)

        for i in range(L):
            wordvec = tf.gather_nd(word_embeddings, [[sentence_id_list[i]]])
            output, state = rnncell(wordvec, state)

        generate_length = 0
        for i in range(L, max_length):
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
        print (' '.join(sentence[:L+generate_length]))
        result_ptr.write(' '.join(sentence[:L+generate_length])+'\n')

if __name__ == '__main__':

    train_path = "data/sentences.train"
    train_text = load_data(train_path)
    word2id_dict, id2word_dict = build_dict(train_text,vocab_len=vocab_len)
    train_data = convert_text_data(train_text[:text_num], word2id_dict)
    data_x, data_y, mask = get_batch_data(train_data, batch_size=batch_size)
    embedding_path = "wordembeddings-dim100.word2vec"

    # # Training
    # sess = tf.Session()
    # with sess.as_default():
    #     model = RNNmodel(vocab_len=vocab_len,
    #                      embedding_size=embedding_dim,
    #                      hidden_size=hidden_size,
    #                      sequency_length=max_length,
    #                      batch_size=batch_size,
    #                      is_add_layer=is_add_layer)
    #
    #     # define the training process
    #     tvars = tf.trainable_variables()
    #     grads, _ = tf.clip_by_global_norm(tf.gradients(model.minimize_loss, tvars), max_grad_norm)
    #     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #     global_step = tf.Variable(0, name="global_step", trainable=False)
    #     train_op = optimizer.apply_gradients(zip(grads, tvars),
    #         global_step=global_step)
    #     sess.run(tf.global_variables_initializer())
    #
    #     #  add summary
    #     grad_summaries = []
    #     for g, v in zip(grads,tvars):
    #         if g is not None:
    #             grad_hist_summary = tf.summary.histogram('/grad/hist/%s' % v.name, g)
    #             grad_summaries.append(grad_hist_summary)
    #     grad_summaries_merged = tf.summary.merge(grad_summaries)
    #
    #     # set the output dir
    #     timestamp = str(int(time.time()))
    #     out_dir = os.path.abspath(os.path.join(os.path.curdir, 'run', timestamp))
    #     print("write to {}\n".format(out_dir),flush=True)
    #
    #     # summary for the loss
    #     loss_summary = tf.summary.scalar("print_perplexity", model.print_perplexity)
    #
    #     # train_summary
    #     train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    #     out_summary_dir = os.path.join(out_dir, "summary")
    #     train_summary_writer = tf.summary.FileWriter(out_summary_dir,sess.graph)
    #
    #     # saver
    #     checkpoint_dir = os.path.join(out_dir,"checkpoints")
    #     if not os.path.exists(checkpoint_dir):
    #         os.makedirs(checkpoint_dir)
    #     saver = tf.train.Saver(tf.global_variables())
    #
    #     # load embedding
    #     wordemb = gensim.models.KeyedVectors.load_word2vec_format(embedding_path,binary=False)
    #     my_embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_len,embedding_dim))
    #     if is_use_embedding:
    #         for id, word in id2word_dict.items():
    #             if word in wordemb.vocab:
    #                 my_embedding_matrix[id,:] = wordemb[word]
    #             else:
    #                 my_embedding_matrix[id,:] = np.random.uniform(-0.25, 0.25,embedding_dim)
    #
    #     word_embedding = tf.placeholder(tf.float32,[None,None], name="pretrained_embeddings")
    #     set_x = model.word_embeddings.assign(word_embedding)
    #
    #     sess.run(set_x, feed_dict={word_embedding:my_embedding_matrix})
    #
    #     epoch = 0
    #     while epoch < num_epoch:
    #         batch_loss = 0
    #         for ibatch in range(data_x.shape[0]):
    #              _, summary, step, loss = sess.run([train_op, train_summary_op, global_step, model.print_perplexity], feed_dict={model.input_x: data_x[ibatch,:,:],
    #                                                                      model.input_y: data_y[ibatch,:,:],
    #                                                                      model.sequence_length_list: mask[ibatch,:]})
    #              train_summary_writer.add_summary(summary=summary, global_step=step)
    #              print(ibatch, loss)
    #              batch_loss += loss
    #         print("ipoch", epoch, "loss", batch_loss/data_x.shape[0])
    #         sys.stdout.flush()
    #         epoch += 1
    #     saver.save(sess,checkpoint_dir,global_step=step)
    #
    #
    #
    # valid_path = "data/sentences.eval"
    # valid_text = load_data(valid_path)
    # all_valid_data = convert_text_data(valid_text, word2id_dict)
    # vdata_x, vdata_y, vsequence_mask = get_batch_data(all_valid_data, batch_size=batch_size)
    # result_path = out_dir
    # pepfile_path = os.path.join(result_path,'perplexity.txt')
    # result_ptr = open(pepfile_path, 'w')
    #
    # evaluate(sess_path=result_path,
    #          data_x=vdata_x,
    #          data_y=vdata_y,
    #          length_list=vsequence_mask,
    #          result_ptr=result_ptr)

    cont_path = "data/sentences.continuation"
    cont_text = load_data(cont_path)
    all_cont_data = convert_text_data(cont_text, word2id_dict)
    cdata_x, cdata_y, csequence_rnnmask = get_batch_data(all_cont_data, batch_size=1)
    result_path = '/Users/jiayu/PycharmProjects/NLU/run/1523817047'
    pepfile_path = os.path.join(result_path,'continuation.txt')
    result_ptr = open(pepfile_path, 'w')

    generate(sess_path=result_path,
             data_x=cdata_x,
             length_list=csequence_rnnmask,
             result_ptr=result_ptr,
             is_add_layer=is_add_layer)



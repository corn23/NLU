# Neural Language Model
import numpy as np
import tensorflow as tf
from RNN import RNNmodel
import gensim
import time
import os

vocab_len = 1000
num_epoch = 6
max_length = 30
batch_size = 50
embedding_dim = 100
hidden_size = 512
max_grad_norm = 5
text_num = 1000  # for quick dry-run
learing_rate = 0.1

def load_data(path):
    f = open(path,'r')
    data = []
    for line in f:
        data.append(line.split(' ')[:-1])
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
    if len(IDlist) > 28:
        IDlist = IDlist[:28]
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


train_path = "data/sentences.eval"
data = load_data(train_path)
word2id_dict, id2word_dict = build_dict(data,vocab_len=vocab_len)
all_train_data = convert_text_data(data[:text_num], word2id_dict)
data_x, data_y, mask = get_batch_data(all_train_data, batch_size=batch_size)
embedding_path = "wordembeddings-dim100.word2vec"



# Training
sess = tf.Session()
with sess.as_default():
    model = RNNmodel(vocab_len=vocab_len,
                     embedding_size=embedding_dim,
                     hidden_size=hidden_size,
                     sequency_length=max_length,
                     batch_size=batch_size)

    # define the training process
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, tvars), max_grad_norm)
    optimizer = tf.train.AdamOptimizer(learning_rate=learing_rate)
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
    out_dir = os.path.abspath(os.path.join(os.path.curdir,'run', timestamp))
    print("write to {}\n".format(out_dir))

    # summary for the loss
    loss_summary = tf.summary.scalar("loss", model.loss)

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

    for id, word in id2word_dict.items():
        if word in wordemb.vocab:
            my_embedding_matrix[id,:] = wordemb[word]
        else:
            my_embedding_matrix[id,:] = np.random.uniform(-0.25, 0.25,embedding_dim)

    word_embedding = tf.placeholder(tf.float32,[None,None], name="pretrained_embeddings")
    set_x = model.word_embeddings.assign(word_embedding)

    sess.run(set_x, feed_dict={word_embedding:my_embedding_matrix})

    epoch = 0
    while epoch < num_epoch:
        batch_loss = 0
        for ibatch in range(data_x.shape[0]):
             _, summary, step, loss = sess.run([train_op, train_summary_op, global_step, model.perplexity], feed_dict={model.input_x: data_x[ibatch,:,:],
                                                                     model.input_y: data_y[ibatch,:,:],
                                                                     model.sequence_length_list: mask[ibatch,:]})
             train_summary_writer.add_summary(summary=summary, global_step=step)
             batch_loss += loss
        print("ipoch", epoch, "loss", batch_loss/data_x.shape[0])
        epoch += 1
    saver.save(sess,checkpoint_dir,global_step=step)

#
# evaluation phase
valid_path = "data/sentences.eval"
data = load_data(valid_path)
all_valid_data = convert_text_data(data, word2id_dict)
data_x, data_y, sequemce_mask = all_valid_data


def evaluate(sess_path, sentence, target, length_list, my_embedding_matrix):
    sess = tf.Session()
    graph_name = sess_path + '.meta'
    saver = tf.train.import_meta_graph(graph_name)
    saver.restore(sess, sess_path)
    graph = tf.get_default_graph()

    # get ops
    perplexity = graph.get_tensor_by_name("perplexity:0")
    input_x = graph.get_tensor_by_name("input_x:0")
    input_y = graph.get_tensor_by_name("input_y:0")
    sequence_length_list = graph.get_tensor_by_name("sequence_length_list:0")

    this_perplexity = sess.run(perplexity, feed_dict={input_x:data_x[:50],
                                            input_y:data_y[:50],
                                            sequence_length_list:sequemce_mask[:50]})

    return perplexity

def predict(sess, sentence, mask, model):
    pred_id = sess.run(model.prediction, feed_dict={model.input:sentence})
    next_word_id = pred_id[mask]
    next_word = id2word(next_word_id)

    return next_word

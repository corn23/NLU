import numpy as np


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


def convert_text_data(data, word2id_dict,max_length):
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


def get_batch_data(text, word2id_dict, batch_size,max_length):
    data_x, data_y, mask = convert_text_data(text,word2id_dict,max_length)
    num_batch = len(data_x)//batch_size
    data_x = np.reshape(data_x[:num_batch*batch_size,:], [num_batch, batch_size, max_length])
    data_y = np.reshape(data_y[:num_batch*batch_size,:], [num_batch, batch_size, max_length])
    mask = np.reshape(mask[:num_batch*batch_size], [num_batch, batch_size])
    return data_x, data_y, mask

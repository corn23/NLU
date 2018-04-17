# Neural Language Model

import os
import pickle
import sys

import tensorflow as tf
from RNN import RNNmodel
from utils import load_data,build_dict,get_batch_data
from model import train,evaluate,generate
from get_cfg import get_cfg

cfg = get_cfg()


print("vocab_len:{} num_epch:{} text_num:{} learning_rate:{}".format(
cfg['vocab_len'], cfg['num_epoch'],cfg['text_num'],cfg['learning_rate']))
print("is_use_embedding",cfg['is_use_embedding'])
print("is_add_layer",cfg['is_add_layer'])

# load dict
with open('word2id_dict.pkl', 'rb') as f:
    word2id_dict = pickle.load(f)
id2word_dict={}
for item in word2id_dict.items():
    id2word_dict[item[1]] = item[0]

if cfg['t']:
    train_path = "data/sentences.train"
    train_text = load_data(train_path)
    word2id_dict, id2word_dict = build_dict(train_text,vocab_len=cfg['vocab_len'])
    train_batch_data = get_batch_data(train_text, word2id_dict =word2id_dict,
                                      batch_size=cfg['batch_size'], max_length=cfg['max_length'])

    model = RNNmodel(vocab_len=cfg['vocab_len'],
                     embedding_size=cfg['embedding_dim'],
                     hidden_size=cfg['hidden_size'],
                     sequency_length=cfg['max_length'],
                     batch_size=cfg['batch_size'],
                     is_add_layer=cfg['is_add_layer'])

    result_path = train(model=model,cfg=cfg,
                    id2word_dict=id2word_dict,train_data=train_batch_data)


if cfg['e']:
    valid_path = "data/sentences.eval"
    valid_text = load_data(valid_path)
    if not cfg['t']:
        if len(cfg['sess_path'])==0:
            print('no model sess path specified. Can not preceed.')
            sys.exit(1)
        else:
            result_path=cfg['sess_path']

    valid_batch_data = get_batch_data(valid_text,word2id_dict=word2id_dict,
                                      batch_size=cfg['batch_size'],max_length=cfg['max_length'])

    pepfile_path = os.path.join(result_path,'perplexity.txt')
    result_ptr = open(pepfile_path, 'w')

    print("start evaluate the language model")
    evaluate(sess_path=result_path,
             eva_data=valid_batch_data,
             result_ptr=result_ptr)
    print("evaluation phase completed")

if cfg['g']:
    if not cfg['t']:
        if len(cfg['sess_path'])==0:
            print('no model sess path specified. Can not preceed.')
            sys.exit(1)
        else:
            result_path=cfg['sess_path']

    # get id2word_dict
    tf.reset_default_graph()
    contin_path = "data/sentences.continuation"
    contin_text = load_data(contin_path)
    contin_batch_data = get_batch_data(contin_text, word2id_dict=word2id_dict,
                                        batch_size=1, max_length=cfg['max_length'])
    pepfile_path = os.path.join(result_path,'continuation.txt')
    result_ptr = open(pepfile_path, 'w')

    print("start generating phase")
    generate(sess_path=result_path,
             contin_data=contin_batch_data,
             result_ptr=result_ptr,
             id2word_dict=id2word_dict,
             cfg=cfg)
    print("generating phase completed")


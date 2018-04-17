import argparse
import sys

def get_cfg():
    parser = argparse.ArgumentParser(description="NLU project 1")
    parser.add_argument('-t', type=bool, help='train phase flag. True means start train.Default False', default=False)
    parser.add_argument('-e', type=bool, help='evaluate phase flag. True means start train.Default False', default=False)
    parser.add_argument('-g', type=bool, help='continuation phase flag. True means start train.Default False', default=False)
    parser.add_argument('-sess_path',type=str,help='specify the model path.Default empty',default='')
    parser.add_argument('-num_epoch',type=int,help='specify the training epoch in traing phase.Default 1',default=1)
    parser.add_argument('-batch_size',type=int,help='specify the batch size of traing phase.Default 100',default=100)
    parser.add_argument('-max_generate_length',type=int,help='specify the maximum generating length in generating phase.Default 20', default=20)
    parser.add_argument('-learning_rate',type=float,help="specify the learning rate in training phase.Default 0.01", default=0.01)
    parser.add_argument('-max_grad_norm',type=float,help="specify the maximu grad norm in training phase.Default 5", default=5)
    parser.add_argument('-train_set_size',type=int,help="specify the training set size.Default 200000",default=200000)
    parser.add_argument('-max_length', type=int,help='specify the maximum length we want to deal with.Default 30', default=30)
    parser.add_argument('-is_add_layer',type=bool,help="indicate if additional layer added in training and generating phase.Default False",default=False)
    parser.add_argument('-is_use_embedding',type=bool,help='indicate if pre-trained word embedding is used in training phase.Default False',default=False)

    cfg = {}
    args = parser.parse_args()

    if len(sys.argv) < 3:
        print("no mode is specified, please at least choose one mode you among [-t, -e -g]")
        print("program will eixt")
        parser.print_help()
        sys.exit(1)

    cfg['t'] = args.t
    cfg['e'] = args.e
    cfg['g'] = args.g
    cfg['sess_path'] = args.sess_path
    cfg['num_epoch'] = args.num_epoch
    cfg['max_length'] = args.max_length
    cfg['batch_size'] = args.batch_size
    cfg['max_generate_length']=args.max_generate_length
    cfg['max_grad_norm'] = args.max_grad_norm
    cfg['text_num'] = args.train_set_size
    cfg['learning_rate'] = args.learning_rate
    cfg['is_add_layer'] = args.is_add_layer
    cfg['is_use_embedding'] = args.is_use_embedding

    # default parameter
    cfg['hidden_size'] = 512
    cfg['embedding_path'] = "wordembeddings-dim100.word2vec"
    cfg['embedding_dim'] = 100
    cfg['vocab_len'] = 20000

    return cfg

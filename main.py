import os
import numpy as np
import dynet as dy

def ptb(section='test.txt', directory='ptb/', padding='<EOS>', column=0):
    with open(os.path.join(directory, section), 'rt') as fh:
        data = [ i.split('\t')[column] for i in fh ]
    data = [ ex.strip().split(' ') for ex in data ]
    data = [ ex + [padding] for ex in data ]
    vocab = set([ word for sent in data for word in sent ])
    return vocab, data

def read_vocab(vocab='vocab', directory='data/'):
    with open(os.path.join(directory, vocab), 'rt') as fh:
        vocab = [ i.strip().split('\t')[0] for i in fh ]
    return vocab

def text_to_sequence(texts, vocab, maxlen=30, padding='<EOS>', mask=0.):
    word_to_n = { word : i for i, word in enumerate(vocab, 1) }
    n_to_word = { i : word for word, i in word_to_n.items() }

    sequences = []
    for sent in texts:
        sequences.append([ word_to_n[word] for word in sent ])
    return sequences, word_to_n, n_to_word

def sort_by_len(X_train, y_train):
    data = list(zip(X_train, y_train))
    data.sort(key=lambda x: len(X_train))
    return [ i[0] for i in data ], [ i[1] for i in data ]

class Seq2SeqAttention:
    def __init__(self, parameters, vocab_size, out_vocab_size, embedding_dim=128, encoder_layers=3, decoder_layers=3, \
            encoder_hidden_dim=256, decoder_hidden_dim=256, encoder_dropout=0.5, decoder_dropout=0.5):
        self.collection = colllection
        self.params = {}

        #self.params['W_emb'] =
        encoder = [ dy.LSTMBuilder(encoder_layers, embedding_dim, encoder_hidden_dim, parameters), \
                dy.LSTMBuilder(decoder_layers, embedding_dim, decoder_hidden_dim, parameters) ]
        decoder = [ dy.LSTMBuilder(1, encoder_hidden_dim, decoder_hidden_dim, parameters), \
                dy.LSTMBuilder(decoder_layers-1, embedding_dim, decoder_hidden_dim, parameters) ]

    def one_sequence(X, y):
        pass

if __name__ == '__main__':
    print('Reading vocab...')
    in_vocab = read_vocab()
    in_vocab +=  ['<unk>', '<EOS>']
    out_vocab = ['<mask>', '<EOS>', '(', ')', '<TOK>']
    print('Done.')

    print('Reading train/valid data...')
    _, X_train = ptb(section='wsj_23', directory='data/', column=0)
    _, y_train = ptb(section='wsj_23', directory='data/', column=1)
    X_train_seq, word_to_n, n_to_word = text_to_sequence(X_train, in_vocab, maxlen=50)
    y_train_seq, _, _ = text_to_sequence(y_train, out_vocab, maxlen=50, mask=1.)

    _, X_valid = ptb(section='wsj_24', directory='data/', column=0)
    _, y_valid = ptb(section='wsj_24', directory='data/', column=1)
    X_valid_seq, word_to_n, _ = text_to_sequence(X_valid, in_vocab, maxlen=50)
    y_valid_seq, _, _ = text_to_sequence(y_valid, out_vocab, maxlen=50, mask=1.)
    print('Done.')

    print('Contains %d unique words.' % len(in_vocab))
    print('Read in %d examples.' % len(X_train))

    print('Checkpointing models on validation loss...')
    RUN = 'runs/baseline'
    checkpoint = os.path.join(RUN, 'baseline.h5')
    print('Checkpoints will be written to %s.' % checkpoint)

    print('Building model...')
    collection = dy.ParameterCollection()
    seq2seq = Seq2SeqAttention(collection, len(in_vocab), len(out_vocab))
    print('Done.')

    #print('Loading last model...')
    #print('Done.')

    print('Training model...')
    EPOCHS = 1000
    BATCH_SIZE = 128
    for epochs in range(0, EPOCHS):
        pass

    print('Done.')


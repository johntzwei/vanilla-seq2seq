from __future__ import print_function

import os
import sys
import time
import random
import numpy as np
import dynet as dy

random.seed(0)

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

def text_to_sequence(texts, vocab, maxlen=30, padding='<EOS>'):
    word_to_n = { word : i for i, word in enumerate(vocab, 0) }
    n_to_word = { i : word for word, i in word_to_n.items() }

    sequences = []
    for sent in texts:
        sequences.append([ word_to_n[word] for word in sent ])
    return sequences, word_to_n, n_to_word

def sort_by_len(X, y):
    data = list(zip(X, y))
    data.sort(key=lambda x: -len(x[1]))
    return [ i[0] for i in data ], [ i[1] for i in data ]

class Seq2SeqAttention:
    def __init__(self, collection, vocab_size, out_vocab_size, embedding_dim=128, encoder_layers=3, decoder_layers=3, \
            encoder_hidden_dim=256, decoder_hidden_dim=256, encoder_dropout=0.5, decoder_dropout=0.5):
        self.collection = collection
        self.params = {}

        self.params['W_emb'] = collection.add_lookup_parameters((embedding_dim, vocab_size))
        self.encoder = [ dy.LSTMBuilder(encoder_layers, embedding_dim, encoder_hidden_dim, collection), \
                dy.LSTMBuilder(decoder_layers, embedding_dim, decoder_hidden_dim, collection) ]

        self.decoder = [ dy.LSTMBuilder(1, encoder_hidden_dim, decoder_hidden_dim, collection), \
                dy.LSTMBuilder(decoder_layers-1, decoder_hidden_dim, decoder_hidden_dim, collection) ]
        self.params['W_1'] = collection.add_parameters((decoder_hidden_dim, encoder_hidden_dim)) 
        self.params['W_2'] = collection.add_parameters((decoder_hidden_dim, decoder_hidden_dim)) 
        self.params['vT'] = collection.add_parameters((1, decoder_hidden_dim,)) 

        self.params['W_o'] = collection.add_parameters((out_vocab_size, decoder_hidden_dim)) 
        self.params['b_o'] = collection.add_parameters((out_vocab_size,)) 

    def load_params(self):
        self.batch_params = (
            dy.parameter(self.params['W_emb']), \
            dy.parameter(self.params['W_1']), \
            dy.parameter(self.params['W_2']), \
            dy.parameter(self.params['vT']), \
            dy.parameter(self.params['W_o']), \
            dy.parameter(self.params['b_o']), \
        )


    def one_sequence(self, X, maxlen, training=True):
        W_emb, W_1, W_2, vT, W_o, b_o = self.batch_params

        if training:
            self.encoder[0].set_dropouts(0, 0.5)
            self.encoder[1].set_dropouts(0, 0.5)
            self.encoder[0].set_dropouts(0, 0.5)
            self.encoder[1].set_dropouts(0, 0.5)
        else:
            self.encoder[0].set_dropouts(0, 0)
            self.encoder[1].set_dropouts(0, 0)
            self.encoder[0].set_dropouts(0, 0)
            self.encoder[1].set_dropouts(0, 0)

        #encode
        X = [ W_emb[x] for x in X ]
        X_ = X[-1::-1] + X[-1:0]

        lstm = self.encoder[0].initial_state()
        states = lstm.add_inputs(X)
        s1 = states[-1].s()
        forward = [ state.h()[-1] for state in states ]

        lstm = self.encoder[1].initial_state()
        states = lstm.add_inputs(X_)
        s2 = states[-1].s()
        backward = [ state.h()[-1] for state in states ]

        hidden_state = [ x + y for x, y in zip(s1, s2) ]
        encoding = [ x + y for x, y in zip(forward, backward) ]

        #decode
        xs = [ W_1 * h_i for h_i in encoding ]
        encoding = dy.concatenate_cols(encoding)
        c_0, h_0 = hidden_state[0], hidden_state[3]     #dependent on layers
        s0 = self.decoder[0].initial_state(vecs=[c_0, h_0])
        
        hidden = []
        state = s0
        for tok in range(0, maxlen):
            y = W_2 * state.h()[-1]
            u = vT * dy.tanh(dy.concatenate_cols([ x + y for x in xs ]))
            a_t = dy.softmax(u)
            d_t = encoding * dy.transpose(a_t)
            
            state = state.add_input(d_t)
            hidden.append(state.h()[-1])

        s0 = self.decoder[1].initial_state(vecs=hidden_state[1:3]+hidden_state[4:])
        hidden = s0.transduce(hidden)

        #logits
        decoding = [ dy.affine_transform([b_o, W_o, h_i]) for h_i in hidden ]
        return decoding

    #takes logits
    def to_sequence(self, decoding, out_vocab):
        decoding = [ dy.softmax(x) for x in decoding ]
        decoding = [ np.argmax(x.value()) for x in decoding ]
        return [ out_vocab[x] for x in decoding ]
        
if __name__ == '__main__':
    print('Reading vocab...')
    in_vocab = read_vocab()
    in_vocab +=  ['<unk>', '<EOS>']
    out_vocab = ['(', ')', '<TOK>', '<EOS>']
    print('Done.')

    print('Reading train/valid data...')
    _, X_train = ptb(section='wsj_2-21', directory='data/', column=0)
    _, y_train = ptb(section='wsj_2-21', directory='data/', column=1)
    X_train_seq, word_to_n, n_to_word = text_to_sequence(X_train, in_vocab)
    y_train_seq, _, _ = text_to_sequence(y_train, out_vocab, maxlen=50)
    X_train_seq, y_train_seq = sort_by_len(X_train_seq, y_train_seq)

    _, X_valid = ptb(section='wsj_24', directory='data/', column=0)
    _, y_valid = ptb(section='wsj_24', directory='data/', column=1)
    X_valid_seq, word_to_n, _ = text_to_sequence(X_valid, in_vocab)
    y_valid_seq, _, _ = text_to_sequence(y_valid, out_vocab, maxlen=50)
    print('Done.')

    print('Contains %d unique words.' % len(in_vocab))
    print('Read in %d examples.' % len(X_train))

    print('Checkpointing models on validation loss...')
    RUN = 'runs/baseline'
    checkpoint = os.path.join(RUN, 'baseline.model')
    print('Checkpoints will be written to %s.' % checkpoint)

    print('Building model...')
    collection = dy.ParameterCollection()
    seq2seq = Seq2SeqAttention(collection, len(in_vocab), len(out_vocab))
    print('Done.')

    #print('Loading last model...')
    #print('Done.')

    print('Training model...')
    EPOCHS = 1000
    BATCH_SIZE = 64
    trainer = dy.AdamTrainer(collection)

    prev_loss = 0.
    for epoch in range(1, EPOCHS+1):
        dy.renew_cg()
        seq2seq.load_params()
        losses = []
        start = time.time()
        loss = 0.

        for i, (X, y) in enumerate(zip(X_train_seq, y_train_seq), 1):
            decoding = seq2seq.one_sequence(X, len(y))
            ex_loss = dy.esum([ dy.pickneglogsoftmax(h, j) for h, j in zip(decoding, y) ])
            losses.append(ex_loss)

            if i == len(X_train_seq) or i % BATCH_SIZE == 0:
                batch_loss = dy.esum(losses)
                batch_loss.backward()
                trainer.update()

                elapsed = time.time() - start
                loss += batch_loss.value()
                avg_batch_loss = loss / (i/BATCH_SIZE)

                dy.renew_cg()
                losses = []
                seq2seq.load_params()
                print('Epoch %d. Time elapsed: %ds, %d/%d. Average batch loss: %f\r' % \
                        (epoch, elapsed, i, len(X_train_seq), avg_batch_loss), end='')

        print()
        print('Done. Total loss: %f' % loss)
        trainer.status()
        print()

        #validation
        print('Validating...')
        loss = 0.
        correct_toks = 0
        total_toks = 0
        validation = open(os.path.join(RUN, 'validation'), 'wt')
        for X_raw, y_raw, X, y in zip(X_valid, y_valid, X_valid_seq, y_valid_seq):
            dy.renew_cg()
            decoding = seq2seq.one_sequence(X, len(y), training=False)
            decoding_seq = seq2seq.to_sequence(decoding, out_vocab)
            validation.write('%s\t%s\t%s\n' % (' '.join(X_raw), ' '.join(y_raw), ' '.join(decoding_seq)))

            ex_loss = dy.esum([ dy.pickneglogsoftmax(h, i) for h, i in zip(decoding, y) ])
            loss += ex_loss.value()

            correct_toks += [ i == j for i, j in zip(decoding_seq, y_raw) ].count(True)
            total_toks += len(y)
        print('Validation loss: %f. Token-level accuracy: %f.' % (loss, correct_toks/total_toks))

        if prev_loss == 0. or loss < prev_loss:
            print('Lowest validation loss yet. Saving model...')
            collection.save(checkpoint)
            prev_loss = loss
    print('Done.')

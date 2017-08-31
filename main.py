from __future__ import print_function

import os
import sys
import time
import random
random.seed(0)

import numpy as np
import dynet_config
dynet_config.set_gpu()
dynet_config.set(mem=8192, random_seed=random.randint(1, 100))
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

def text_to_sequence(texts, vocab):
    word_to_n = { word : i for i, word in enumerate(vocab, 0) }
    n_to_word = { i : word for word, i in word_to_n.items() }
    sequences = []
    for sent in texts:
        sequences.append([ word_to_n[word] for word in sent ])
    return sequences, word_to_n, n_to_word

def sort_by_len(X, y):
    data = list(zip(X, y))
    data.sort(key=lambda x: len(x[1]))
    return [ i[0] for i in data ], [ i[1] for i in data ]

def batch(X, batch_size=128, mask=0.):
    ex, masks = [], []
    for i in xrange(0, len(X), batch_size):
        X_ = X[i:i + batch_size]
        X_len = max([ len(x) for x in X_ ])
        X_padding = [ X_len - len(x) for x in X_ ]
        X_padded = [ x + [mask] * mask_len for x, mask_len  in zip(X_, X_padding) ]
        X_mask = [ [1]*len(x)  + [0]*mask_len for x, mask_len  in zip(X_, X_padding) ]
        ex.append(X_padded)
        masks.append(X_mask)
    return ex, masks

class Seq2SeqAttention:
    def __init__(self, collection, vocab_size, out_vocab_size, embedding_dim=128, encoder_layers=3, decoder_layers=3, \
            encoder_hidden_dim=256, decoder_hidden_dim=256, encoder_dropout=0.5, decoder_dropout=0.5):
        self.collection = collection
        self.params = {}

        self.params['W_emb'] = collection.add_lookup_parameters((vocab_size, embedding_dim))
        self.encoder = [ dy.LSTMBuilder(encoder_layers, embedding_dim, encoder_hidden_dim, collection), \
                dy.LSTMBuilder(encoder_layers, embedding_dim, encoder_hidden_dim, collection) ]

        self.decoder = [ dy.LSTMBuilder(1, encoder_hidden_dim, decoder_hidden_dim, collection), \
                dy.LSTMBuilder(decoder_layers-1, decoder_hidden_dim, decoder_hidden_dim, collection) ]
        self.params['W_1'] = collection.add_parameters((decoder_hidden_dim, encoder_hidden_dim)) 
        self.params['W_2'] = collection.add_parameters((decoder_hidden_dim, decoder_hidden_dim)) 
        self.params['vT'] = collection.add_parameters((1, decoder_hidden_dim,)) 

        self.params['R'] = collection.add_parameters((out_vocab_size, decoder_hidden_dim)) 
        self.params['b'] = collection.add_parameters((out_vocab_size,)) 

    def one_sequence_batch(self, X_batch, maxlen, training=True):
        #params
        W_emb = self.params['W_emb']
        W_1 = dy.parameter(self.params['W_1'])
        W_2 = dy.parameter(self.params['W_2'])
        vT = dy.parameter(self.params['vT'])
        R = dy.parameter(self.params['R'])
        b = dy.parameter(self.params['b'])

        if training:
            self.encoder[0].set_dropouts(0.5, 0)
            self.encoder[1].set_dropouts(0.5, 0)
            self.decoder[0].set_dropouts(0, 0)
            self.decoder[1].set_dropouts(0.5, 0)
        else:
            self.encoder[0].set_dropout(0)
            self.encoder[1].set_dropout(0)
            self.decoder[0].set_dropout(0)
            self.decoder[1].set_dropout(0)

        #encode
        X = [ dy.lookup_batch(self.params['W_emb'], tok_batch) for tok_batch in X_batch ]
        X_ = X[::-1]         #TODO not a real reverse - <mask>, ..., <EOS>, tok_t, tok_t-1, ...

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
        decoding = [ dy.affine_transform([b, R, h_i]) for h_i in hidden ]
        return decoding

    #takes logits
    def to_sequence_batch(self, decoding, out_vocab):
        batch_size = decoding[0].dim()[1]
        decoding = [ dy.softmax(x) for x in decoding ]
        decoding = [ dy.reshape(x, (len(out_vocab), batch_size), batch_size=1) for x in decoding ]
        decoding = [ np.argmax(x.value(), axis=0) for x in decoding ]
        decoding = [  [ x[i] for x in decoding ] for i in range(0, batch_size) ]
        return [ [ out_vocab[y] for y in x ] for x in decoding ]

    def one_batch(self, X_batch, y_batch, masks, training=True):
        batch_size = len(X_batch)
        X_batch = zip(*X_batch)
        y_batch = zip(*y_batch)
        masks = zip(*masks)

        decoding = seq2seq.one_sequence_batch(X_batch, len(y_batch), training=training)
        
        batch_loss = []
        for x, y, mask in zip(decoding, y_batch, masks):
            mask_expr = dy.inputVector(mask)
            mask = dy.reshape(mask_expr, (1,), batch_size)
            batch_loss.append(mask * dy.pickneglogsoftmax_batch(x, y))
        batch_loss = dy.esum(batch_loss)
        batch_loss = dy.sum_batches(batch_loss)

        return batch_loss, decoding
        
if __name__ == '__main__':
    print('Reading vocab...')
    in_vocab = read_vocab()
    in_vocab +=  ['<unk>', '<EOS>', '<mask>']
    out_vocab = ['(', ')', '<TOK>', '<EOS>']
    print('Done.')

    print('Reading train/valid data...')
    BATCH_SIZE = 64
    _, X_train = ptb(section='wsj_2-21', directory='data/', column=0)
    _, y_train = ptb(section='wsj_2-21', directory='data/', column=1)
    X_train, y_train = sort_by_len(X_train, y_train)
    X_train_seq, word_to_n, n_to_word = text_to_sequence(X_train, in_vocab)
    y_train_seq, _, _ = text_to_sequence(y_train, out_vocab)
    X_train_seq, X_train_masks = batch(X_train_seq, batch_size=BATCH_SIZE, mask=len(in_vocab)-1)
    y_train_seq, y_train_masks = batch(y_train_seq, batch_size=BATCH_SIZE, mask=len(in_vocab)-1)

    _, X_valid = ptb(section='wsj_24', directory='data/', column=0)
    _, y_valid = ptb(section='wsj_24', directory='data/', column=1)
    X_valid, y_valid = sort_by_len(X_valid, y_valid)
    X_valid_raw, _ = batch(X_valid, batch_size=BATCH_SIZE, mask='<mask>') 
    y_valid_raw, _ = batch(y_valid, batch_size=BATCH_SIZE, mask='<mask>')

    X_valid_seq, word_to_n, _ = text_to_sequence(X_valid, in_vocab)
    y_valid_seq, _, _ = text_to_sequence(y_valid, out_vocab)
    X_valid_seq, X_valid_masks = batch(X_valid_seq, batch_size=BATCH_SIZE, mask=len(in_vocab)-1) 
    y_valid_seq, y_valid_masks = batch(y_valid_seq, batch_size=BATCH_SIZE, mask=len(in_vocab)-1)
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

    print('Training model...')
    EPOCHS = 1000
    trainer = dy.AdamTrainer(collection)

    lowest_val_loss = 0.
    for epoch in range(1, EPOCHS+1):
        loss = 0.
        start = time.time()

        #learning rate scheduling
        #if epoch > 8:
        #    trainer.learning_rate *= 0.99

        for i, (X_batch, y_batch, masks) in enumerate(zip(X_train_seq, y_train_seq, y_train_masks), 1):
            dy.renew_cg()
            batch_loss, _ = seq2seq.one_batch(X_batch, y_batch, masks)
            batch_loss.backward()
            trainer.update()

            elapsed = time.time() - start
            loss += batch_loss.value()
            avg_batch_loss = loss / i
            ex = min(len(X_train), i * BATCH_SIZE)

            print('Epoch %d. Time elapsed: %ds, %d/%d. Average batch loss: %f\r' % \
                    (epoch, elapsed, ex, len(X_train), avg_batch_loss), end='')

        print()
        print('Done. Total loss: %f' % loss)
        trainer.status()
        print()

        print('Validating...')
        loss = 0.
        correct_toks = 0.
        total_toks = 0.

        validation = open(os.path.join(RUN, 'validation'), 'wt')
        for i, (X_batch, y_batch, masks, X_batch_raw, y_batch_raw) in \
                enumerate(zip(X_valid_seq, y_valid_seq, y_valid_masks, X_valid_raw, y_valid_raw), 1):
            dy.renew_cg()
            batch_loss, decoding = seq2seq.one_batch(X_batch, y_batch, masks, training=False)
            loss += batch_loss.value()

            y_pred = seq2seq.to_sequence_batch(decoding, out_vocab)
            for X_raw, y_raw, y_ in zip(X_batch_raw, y_batch_raw, y_pred):
                validation.write('%s\t%s\t%s\n' % (' '.join(X_raw), ' '.join(y_raw), ' '.join(y_)))
                correct_toks += [ tok_ == tok for tok, tok_ in zip(y_, y_raw) ].count(True)
                total_toks += len(y_)

        print('Validation loss: %f. Token-level accuracy: %f.' % (loss, correct_toks/total_toks))
        validation.close()

        if lowest_val_loss == 0. or loss < lowest_val_loss:
            print('Lowest validation loss yet. Saving model...')
            collection.save(checkpoint)
            lowest_val_loss = loss
        print('Done.')

    print('Done.')

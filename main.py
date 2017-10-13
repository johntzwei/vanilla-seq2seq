from __future__ import print_function

import os
import sys
import time
import random
random.seed(0)

import _gdynet as dy
dy_params = dy.DynetParams()
dy_params.set_random_seed(random.randint(0, 1000))
dy_params.set_autobatch(True)
dy_params.set_requested_gpus(1)
dy_params.set_mem(20480)
dy_params.init()

import numpy as np

from model import Seq2SeqAttention

def ptb(section='test.txt', directory='ptb/', padding='<EOS>', column=0, TOK=True):
    with open(os.path.join(directory, section), 'rt') as fh:
        data = [ i.split('\t')[column] for i in fh ]
    data = [ ex.strip().split(' ') for ex in data ]
    data = [ ex + [padding] for ex in data ]

    if TOK:
        data = [ [ tok for tok in ex if tok != '<TOK>' ] for ex in data ]
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

def batch(X, batch_size, mask=0.):
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

if __name__ == '__main__':
    print('Reading vocab...')
    in_vocab = read_vocab()
    in_vocab +=  ['<unk>', '<EOS>', '<mask>']
    out_vocab = read_vocab(vocab='out_vocab')
    out_vocab += ['<EOS>', '<mask>']
    print('Done.')

    print('Reading train/valid data...')
    BATCH_SIZE = 128
    VALID_BATCH_SIZE = 32
    _, X_train = ptb(section='wsj_2-21', directory='data/', column=0)
    _, y_train = ptb(section='wsj_2-21', directory='data/', column=1)
    X_train, y_train = sort_by_len(X_train, y_train)
    X_train_long, y_train_long = X_train[-300:-30], y_train[-300:-30] 
    X_train_short, y_train_short = X_train[:-300], y_train[:-300]
    X_train_seq, word_to_n, n_to_word = text_to_sequence(X_train_short, in_vocab)
    y_train_seq, _, _ = text_to_sequence(y_train_short, out_vocab)
    X_train_long_seq, _, _ = text_to_sequence(X_train_long, in_vocab)
    y_train_long_seq, _, _ = text_to_sequence(y_train_long, out_vocab)
    X_train_seq, X_train_masks = batch(X_train_seq, batch_size=BATCH_SIZE, mask=len(in_vocab)-1)
    y_train_seq, y_train_masks = batch(y_train_seq, batch_size=BATCH_SIZE, mask=len(out_vocab)-1)
    X_train_long_seq, X_train_long_masks = batch(X_train_long_seq, batch_size=32, mask=len(in_vocab)-1)
    y_train_long_seq, y_train_long_masks = batch(y_train_long_seq, batch_size=32, mask=len(out_vocab)-1)

    X_train_seq = X_train_seq + X_train_long_seq
    y_train_seq = y_train_seq + y_train_long_seq
    X_train_masks = X_train_masks + X_train_long_masks 
    y_train_masks = y_train_masks + y_train_long_masks

    _, X_valid = ptb(section='wsj_24', directory='data/', column=0)
    _, y_valid = ptb(section='wsj_24', directory='data/', column=1)
    X_valid, y_valid = sort_by_len(X_valid, y_valid)
    X_valid_raw, _ = batch(X_valid, batch_size=VALID_BATCH_SIZE, mask='<mask>') 
    y_valid_raw, _ = batch(y_valid, batch_size=VALID_BATCH_SIZE, mask='<mask>')

    X_valid_seq, word_to_n, _ = text_to_sequence(X_valid, in_vocab)
    y_valid_seq, _, _ = text_to_sequence(y_valid, out_vocab)
    X_valid_seq, X_valid_masks = batch(X_valid_seq, batch_size=VALID_BATCH_SIZE, mask=len(in_vocab)-1) 
    y_valid_seq, y_valid_masks = batch(y_valid_seq, batch_size=VALID_BATCH_SIZE, mask=len(out_vocab)-1)
    print('Done.')

    print('Contains %d unique words.' % len(in_vocab))
    print('Read in %d examples.' % len(X_train))

    print('Input vocabulary sample...')
    print('\n'.join(in_vocab[:10]))
    print('Output vocabulary sample...')
    print('\n'.join(out_vocab[:10]))

    i = random.randint(0, len(X_train)-1)
    print('Showing example %d out of %d' % (i, len(X_train)))
    print(' '.join(X_train[i]))
    print(' '.join(y_train[i]))

    print('Checkpointing models on validation loss...')
    lowest_val_loss = 0.
    highest_val_accuracy = 0.
    last_train_loss = 0.

    RUN = 'runs/baseline'
    checkpoint = os.path.join(RUN, 'baseline.model')
    print('Checkpoints will be written to %s.' % checkpoint)

    print('Building model...')
    collection = dy.ParameterCollection()
    seq2seq = Seq2SeqAttention(collection, len(in_vocab), len(out_vocab))
    print('Done.')

    #print('Loading model...')
    #collection.populate(checkpoint)
    #print('Done.')

    print('Training model...')
    EPOCHS = 3000
    epochs_to_update = 1
    patience = 0
    e0 = 0.005
    trainer = dy.AdamTrainer(collection, alpha=e0, beta_1=0.85, beta_2=0.997, eps=1e-8, edecay=1)
    trainer.set_clip_threshold(10.0)

    for epoch in range(1, EPOCHS+1):
        loss = 0.
        start = time.time()

        for i, (X_batch, y_batch, X_masks, y_masks) in \
                enumerate(zip(X_train_seq, y_train_seq, X_train_masks, y_train_masks), 1):
            dy.renew_cg()
            batch_loss, _ = seq2seq.one_batch(X_batch, y_batch, X_masks, y_masks)
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
        for i, (X_batch, y_batch, X_masks, y_masks, X_batch_raw, y_batch_raw) in \
                enumerate(zip(X_valid_seq, y_valid_seq, X_valid_masks, y_valid_masks, X_valid_raw, y_valid_raw), 1):
            dy.renew_cg()
            batch_loss, decoding = seq2seq.one_batch(X_batch, y_batch, X_masks, y_masks, training=False)
            loss += batch_loss.value()

            y_pred = seq2seq.to_sequence_batch(decoding, out_vocab)
            for X_raw, y_raw, y_ in zip(X_batch_raw, y_batch_raw, y_pred):
                validation.write('%s\t%s\t%s\n' % (' '.join(X_raw), ' '.join(y_raw), ' '.join(y_)))
                correct_toks += [ tok_ == tok for tok, tok_ in zip(y_, y_raw) ].count(True)
                total_toks += len(y_)
        validation.close()

        accuracy = correct_toks/total_toks
        print('Validation loss: %f. Token-level accuracy: %f.' % (loss, accuracy))

        save = False
        if lowest_val_loss == 0. or loss < lowest_val_loss:
            print('Lowest validation loss yet.')
            save = True
            lowest_val_loss = loss

        if highest_val_accuracy == 0. or accuracy > highest_val_accuracy:
            print('Highest accuracy yet.')
            save = True
            highest_val_accuracy = accuracy

        if last_train_loss == 0 or avg_batch_loss < last_train_loss:
            print('Lowest training loss yet.')
            save = True
            lowest_train_loss = avg_batch_loss
        else:
            if patience > 1:
                print('No improvement. Actually halving learning rate.')
                trainer.update_epoch(epochs_to_update)
                epochs_to_update *= 2
                patience = 0
            else:
                print('Patience at %d' % patience)
                patience += 1
        last_train_loss = avg_batch_loss

        if save:
            print('Saving model...')
            collection.save(checkpoint)

        print('Done.')
    print('Done.')

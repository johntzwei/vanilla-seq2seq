import os
import random
random.seed(0)

from model import Seq2SeqAttention
from utils import ptb, read_vocab, text_to_sequence, \
        sort_by_len, batch

import _gdynet as dy

if __name__ == '__main__':
    print('Reading vocab...')
    in_vocab = read_vocab()
    in_vocab +=  ['<unk>', '<EOS>', '<mask>']
    out_vocab = read_vocab(vocab='out_vocab')
    out_vocab += ['<EOS>', '<mask>'] 
    print('Done.')

    print('Reading test data...')
    BATCH_SIZE = 32
    _, X_test = ptb(section='wsj_23', directory='data/', column=0)
    _, y_test = ptb(section='wsj_23', directory='data/', column=1)
    X_test, y_test = sort_by_len(X_test, y_test)
    X_test_raw, _ = batch(X_test, batch_size=BATCH_SIZE, mask='<mask>') 
    y_test_raw, _ = batch(y_test, batch_size=BATCH_SIZE, mask='<mask>')
    X_test_seq, word_to_n, n_to_word = text_to_sequence(X_test, in_vocab)
    y_test_seq, _, _ = text_to_sequence(y_test, out_vocab)
    X_test_seq, X_test_masks = batch(X_test_seq, batch_size=BATCH_SIZE, mask=len(in_vocab)-1)
    y_test_seq, y_test_masks = batch(y_test_seq, batch_size=BATCH_SIZE, mask=len(in_vocab)-1)
    print('Done.')

    print('Building model...')
    collection = dy.ParameterCollection()
    seq2seq = Seq2SeqAttention(collection, len(in_vocab), len(out_vocab))
    print('Done.')

    print('Loading model...')
    RUN = 'runs/baseline'
    checkpoint = os.path.join(RUN, 'baseline.model')
    print('Loading from %s.' % checkpoint)
    collection.populate(checkpoint)
    print('Done.')

    print('Testing...')
    loss = 0.
    correct_toks = 0.
    total_toks = 0.

    test = open(os.path.join(RUN, 'test'), 'wt')
    for i, (X_batch, y_batch, X_masks, y_masks, X_batch_raw, y_batch_raw) in \
            enumerate(zip(X_test_seq, y_test_seq, X_test_masks, y_test_masks, X_test_raw, y_test_raw), 1):
        dy.renew_cg()
        batch_loss, decoding = seq2seq.one_batch(X_batch, y_batch, X_masks, y_masks, training=False)
        loss += batch_loss.value()

        y_pred = seq2seq.to_sequence_batch(decoding, out_vocab)
        for X_raw, y_raw, y_ in zip(X_batch_raw, y_batch_raw, y_pred):
            test.write('%s\t%s\t%s\n' % (' '.join(X_raw), ' '.join(y_raw), ' '.join(y_)))
            y_raw = y_raw if '<EOS>' not in y_raw else y_raw[:y_raw.index('<EOS>')]
            correct_toks += [ tok_ == tok for tok, tok_ in zip(y_, y_raw) ].count(True)
            total_toks += len(y_)
    test.close()

    accuracy = correct_toks/total_toks
    print('Testing loss: %f. Token-level accuracy: %f.' % (loss, accuracy))
    print('Done.')

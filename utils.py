import os

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

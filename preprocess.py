import re
from nltk.corpus import BracketParseCorpusReader

def normalize(word):
    try:
        if float(word):
            return 'N'
    except:
        pass
    word = word.lower()
    return word

def linearize(tree, label=False, token=False, margin=1000):
    if not label:
        tree.set_label('')
        for subtree in tree.subtrees():
            subtree.set_label('')
    else:
        def norm(label):
            punct = ['``', "''" , '.', ',', ':']
            if label in punct:
                return 'PUNCT'
            
            if label == '$':
                return label

            if label == '-RRB-' or label == '-LRB-' or label == '-NONE-':
                return label

            chars = ['$', '|', '-', '=']
            for char in chars:
                if char in label:
                    label = label[:label.index(char)]

            return label

        label = tree.label()
        tree.set_label(norm(label))

        for subtree in tree.subtrees():
            label = norm(subtree.label())
            subtree.set_label(label)

    for subtree in tree.subtrees(filter=lambda x: x.height() == 2):
        leaf = normalize(subtree[0])
        
        if token:
            subtree[0] = '<TOK>'
            continue

        if leaf not in vocab:
            subtree[0] = '<unk>'
        else:
            subtree[0] = leaf

    #output as string
    lin = tree.pformat(margin=margin, nodesep='', parens=['(', ' )'])
    lin = re.sub(r'\s+', ' ', lin)

    return lin

def label_closing_brackets(lin):
    stack = []
    lin = lin.split(' ')
    for i, tok in enumerate(lin):
        if tok.startswith('('):
            stack.append(tok)

        if tok == ')':
            lin[i] = tok + stack.pop()[1:]

    return ' '.join(lin)

def get_vocab(fn='data/vocab', symbols=2):
    vocab = {}
    for sections in SECTIONS:
        for section in range(sections[0], sections[1]+1):
            fileids = [ i for i in ptb.fileids() if i.startswith(str(section).zfill(2)) ]

            for sent in ptb.sents(fileids):
                for word in sent:
                    word = normalize(word)
                    counter = vocab.get(word, 0)
                    vocab[word] = counter + 1

    #filter out vocab
    vocab = list(vocab.items())
    vocab.sort(key=lambda x: -x[1])
    vocab = vocab[:10000-symbols]

    h = open(fn, 'wt')
    for word, freq in vocab:
        h.write('%s\n' % word)
    h.close()

    vocab = [ i[0] for i in vocab ]
    return vocab

def get_out_vocab(out='data/out_vocab'):
    out_vocab = set()
    for fn, sections in zip([ TRAIN_FILE, TEST_FILE, DEV_FILE ], SECTIONS):
        for line in open(fn, 'rt'):
            tree = line.split('\t')[-1]
            toks = tree.split()
            [ out_vocab.add(tok) for tok in toks ]
    out_vocab = list(out_vocab)

    h = open(out, 'wt')
    for tok in out_vocab:
        h.write('%s\n' % tok)

    return out_vocab

if __name__ == '__main__':
    TRAIN_FILE = 'data/wsj_2-21'
    TEST_FILE = 'data/wsj_23'
    DEV_FILE = 'data/wsj_24'
    SECTIONS = [ (2, 21), (23, 23), (24, 24) ]

    wsj = '/data/penn_tb_3.0/TREEBANK_3/PARSED/MRG/WSJ/'
    file_pattern = r".*/WSJ_.*\.MRG"
    ptb = BracketParseCorpusReader(wsj, file_pattern)
    print('Gathered %d files...' % len(ptb.fileids()))

    print('Generating vocabulary...')
    vocab = get_vocab()
    print('Done.')

    print('Preprocessing all sections...')
    for fn, sections in zip([ TRAIN_FILE, TEST_FILE, DEV_FILE ], SECTIONS):
        print('Preprocessing %s...' % fn)
        h = open(fn, 'wt')
        for section in range(sections[0], sections[1]+1):
            fileids = [ i for i in ptb.fileids() if i.startswith(str(section).zfill(2)) ]
            for sent, tree in zip(ptb.sents(fileids), ptb.parsed_sents(fileids)):
                sent = [ normalize(word) if normalize(word) in vocab else '<unk>' for word in sent ]
                lin = linearize(tree, token=True, label=True)
                lin = label_closing_brackets(lin)
                h.write('%s\t%s\n' % (' '.join(sent), lin))
        h.close()
        print('Done.')
    print('Done.')

    print('Generating output vocabulary...')
    out_vocab = get_out_vocab()
    print('Done.')

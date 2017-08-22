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
                lin = linearize(tree, token=True, label=False)
                h.write('%s\t%s\n' % (' '.join(sent), lin))
        h.close()
        print('Done.')
    print('Done.')

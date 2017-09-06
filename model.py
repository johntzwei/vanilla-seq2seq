import dynet as dy
import numpy as np

class Seq2SeqAttention:
    def __init__(self, collection, vocab_size, out_vocab_size, embedding_dim=128, encoder_layers=3, decoder_layers=3, \
            encoder_hidden_dim=256, decoder_hidden_dim=256, encoder_dropout=0.3, decoder_dropout=0.3, attention_dropout=0.3):
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

        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.attention_dropout = attention_dropout

    def one_sequence_batch(self, X_batch, X_reverse, maxlen, X_masks, training=True):
        #params
        W_emb = self.params['W_emb']
        W_1 = dy.parameter(self.params['W_1'])
        W_2 = dy.parameter(self.params['W_2'])
        vT = dy.parameter(self.params['vT'])
        R = dy.parameter(self.params['R'])
        b = dy.parameter(self.params['b'])

        if training:
            self.encoder[0].set_dropouts(self.encoder_dropout, 0)
            self.encoder[1].set_dropouts(self.encoder_dropout, 0)
            self.decoder[0].set_dropouts(0, 0)
            self.decoder[1].set_dropouts(self.decoder_dropout, 0)
        else:
            self.encoder[0].set_dropouts(0, 0)
            self.encoder[1].set_dropouts(0, 0)
            self.decoder[0].set_dropouts(0, 0)
            self.decoder[1].set_dropouts(0, 0)

        #encode
        X_ = [ dy.lookup_batch(self.params['W_emb'], tok_batch) for tok_batch in X_reverse ]
        X = [ dy.lookup_batch(self.params['W_emb'], tok_batch) for tok_batch in X_batch ]

        lstm = self.encoder[0].initial_state()
        states = lstm.add_inputs(X)
        s1 = states[-1].s()
        forward = [ state.h()[-1] for state in states ]

        lstm = self.encoder[1].initial_state()
        states = lstm.add_inputs(X_)
        s2 = states[-1].s()
        backward = [ state.h()[-1] for state in states ]

        hidden_state = [ x + y for x, y in zip(s1, s2) ]        #hidden state concatenation
        encoding = [ x + y for x, y in zip(forward, backward) ]

        #decode
        xs = [ W_1 * h_i for h_i in encoding ]
        if training:
            xs = [ dy.dropout(x, self.attention_dropout) for x in xs ]

        encoding = dy.concatenate_cols(encoding)
        c_0, h_0 = hidden_state[0], hidden_state[3]     #dependent on layers
        s0 = self.decoder[0].initial_state(vecs=[c_0, h_0])
        
        hidden = []
        state = s0
        for tok in range(0, maxlen):
            y = W_2 * state.h()[-1]
            if training:
                y = dy.dropout(y, self.attention_dropout)

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

    def one_batch(self, X_batch, y_batch, X_masks, y_masks, eos=9999, training=True):
        eoses = [ X.index(eos) for X in X_batch ]
        X_rev = [ x[eos-1::-1] + x[eos:] for x, eos in zip(X_batch, eoses) ]

        batch_size = len(X_batch)
        X_batch = zip(*X_batch)
        X_rev = zip(*X_rev)
        y_batch = zip(*y_batch)
        y_masks = zip(*y_masks)

        decoding = self.one_sequence_batch(X_batch, X_rev, len(y_batch), X_masks, training=training)
        
        batch_loss = []
        for x, y, mask in zip(decoding, y_batch, y_masks):
            mask_expr = dy.inputVector(mask)
            mask = dy.reshape(mask_expr, (1,), batch_size)
            batch_loss.append(mask * dy.pickneglogsoftmax_batch(x, y))
        batch_loss = dy.esum(batch_loss)
        batch_loss = dy.sum_batches(batch_loss)

        return batch_loss, decoding

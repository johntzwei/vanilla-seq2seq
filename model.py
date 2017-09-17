import numpy as np
import _gdynet as dy

class Seq2SeqAttention:
    def __init__(self, collection, vocab_size, out_vocab_size, embedding_dim=128, encoder_layers=3, decoder_layers=3, \
            encoder_hidden_dim=256, decoder_hidden_dim=256, encoder_dropout=0.3, decoder_dropout=0.3, attention_dropout=0.3):
        self.collection = collection
        self.params = {}

        self.params['W_emb'] = collection.add_lookup_parameters((vocab_size, embedding_dim))
        self.params['Wout_emb'] = collection.add_lookup_parameters((out_vocab_size, embedding_dim))
        self.encoder = dy.VanillaLSTMBuilder(encoder_layers, embedding_dim, encoder_hidden_dim, collection)

        self.decoder = [ dy.VanillaLSTMBuilder(decoder_layers-1, embedding_dim, decoder_hidden_dim, collection), \
                dy.VanillaLSTMBuilder(1, encoder_hidden_dim+decoder_hidden_dim, decoder_hidden_dim, collection) ]
        self.params['W_1'] = collection.add_parameters((decoder_hidden_dim, encoder_hidden_dim)) 
        self.params['W_2'] = collection.add_parameters((decoder_hidden_dim, decoder_hidden_dim)) 
        self.params['vT'] = collection.add_parameters((1, decoder_hidden_dim,)) 

        self.params['R'] = collection.add_parameters((out_vocab_size, decoder_hidden_dim)) 
        self.params['b'] = collection.add_parameters((out_vocab_size,)) 

        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.attention_dropout = attention_dropout

    def one_sequence_batch(self, X_batch, maxlen, X_masks, eos=9999, teacher_forcing=None, training=True):
        #params
        W_emb = self.params['W_emb']
        Wout_emb = self.params['Wout_emb']
        W_1 = dy.parameter(self.params['W_1'])
        W_2 = dy.parameter(self.params['W_2'])
        vT = dy.parameter(self.params['vT'])
        R = dy.parameter(self.params['R'])
        b = dy.parameter(self.params['b'])

        if training:
            self.encoder.set_dropouts(self.encoder_dropout, 0)
            self.decoder[0].set_dropouts(0, 0)
            self.decoder[1].set_dropouts(self.decoder_dropout, 0)
        else:
            self.encoder.set_dropouts(0, 0)
            self.decoder[0].set_dropouts(0, 0)
            self.decoder[1].set_dropouts(0, 0)

        #encode
        X = [ dy.lookup_batch(W_emb, tok_batch) for tok_batch in X_batch ]
        lstm = self.encoder.initial_state()
        states = lstm.add_inputs(X)
        s1 = states[-1].s()
        forward = [ state.h()[-1] for state in states ]

        hidden_state = s1
        encoding = forward

        for i, mask in enumerate(X_masks):
            mask_expr = dy.inputVector(mask)
            mask = dy.reshape(mask_expr, (1,), batch_size=len(X_masks[0]))
            encoding[i] = dy.transpose(mask * dy.transpose(encoding[i]))

        encoding = dy.concatenate_cols(encoding)

        #decode
        xs = W_1 * encoding
        if training:
            xs = dy.dropout(xs, self.attention_dropout)

        s0_0 = self.decoder[0].initial_state(vecs=hidden_state[:2]+hidden_state[3:5])
        s0_1 = self.decoder[1].initial_state(vecs=hidden_state[2:3]+hidden_state[5:])

        decoding = []
        state_0 = s0_0.add_input(dy.lookup_batch(self.params['W_emb'], [eos for i in range(0, len(X_batch[0]))]))
        state_1 = s0_1

        for i in range(0, maxlen):
            #attention
            y = W_2 * state_1.h()[-1]
            if training:
                y = dy.dropout(y, self.attention_dropout)

            u = vT * dy.tanh(dy.colwise_add(xs, y))
            a_t = dy.softmax(u)
            d_t = encoding * dy.transpose(a_t)

            #input
            history = state_0.h()[-1]
            inp = dy.concatenate([history, d_t])
            
            #next state
            state_1 = state_1.add_input(inp)
            h_i = state_1.h()[-1]
            decoding.append(dy.softmax(dy.affine_transform([b, R, h_i])))

            if training:
                tf = dy.lookup_batch(Wout_emb, teacher_forcing[i])
                state_0.add_input(tf)
            else:
                print('Not implemented yet!')
        
        #logits
        return decoding

    #takes logits
    def to_sequence_batch(self, decoding, out_vocab):
        batch_size = decoding[0].dim()[1]
        decoding = [ dy.reshape(x, (len(out_vocab), batch_size), batch_size=1) for x in decoding ]
        decoding = [ np.argmax(x.value(), axis=0) for x in decoding ]
        decoding = [  [ x[i] for x in decoding ] for i in range(0, batch_size) ]
        return [ [ out_vocab[y] for y in x ] for x in decoding ]

    def one_batch(self, X_batch, y_batch, X_masks, y_masks, eos=9999, training=True):
        batch_size = len(X_batch)
        X_batch = zip(*X_batch)
        X_masks = zip(*X_masks)
        y_batch = zip(*y_batch)
        y_masks = zip(*y_masks)

        decoding = self.one_sequence_batch(X_batch, len(y_batch), X_masks, \
                eos=eos, teacher_forcing=y_batch, training=training)

        batch_loss = []
        for x, y, mask in zip(decoding, y_batch, y_masks):
            mask_expr = dy.inputVector(mask)
            mask = dy.reshape(mask_expr, (1,), batch_size)
            batch_loss.append(mask * dy.pickneglogsoftmax_batch(x, y))
        batch_loss = dy.esum(batch_loss)
        batch_loss = dy.sum_batches(batch_loss)

        return batch_loss, decoding

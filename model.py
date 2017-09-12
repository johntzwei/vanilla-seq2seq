import dynet as dy
import numpy as np

class VAE_LSTM:
    def __init__(self, collection, vocab_size, out_vocab_size, embedding_dim=128, encoder_layers=3, decoder_layers=3, \
            encoder_hidden_dim=512, decoder_hidden_dim=512, latent_dim=128, encoder_dropout=0., decoder_dropout=0.):
        self.collection = collection
        self.params = {}

        self.params['W_emb'] = collection.add_lookup_parameters((vocab_size, embedding_dim))
        self.encoder = [ dy.LSTMBuilder(encoder_layers, embedding_dim, encoder_hidden_dim, collection), \
                dy.LSTMBuilder(encoder_layers, embedding_dim, encoder_hidden_dim, collection) ]
        self.latent_dim = latent_dim

        self.params['W_mu'] = collection.add_parameters((latent_dim, 2*encoder_hidden_dim)) 
        self.params['W_sigma'] = collection.add_parameters((latent_dim, 2*encoder_hidden_dim)) 

        self.decoder = dy.LSTMBuilder(decoder_layers, embedding_dim, decoder_hidden_dim, collection)
        self.params['R'] = collection.add_parameters((out_vocab_size, decoder_hidden_dim)) 
        self.params['b'] = collection.add_parameters((out_vocab_size,)) 

        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout

    def one_sequence_batch(self, X_batch, X_reverse, maxlen, X_masks, training=True):
        #params
        W_emb = self.params['W_emb']
        W_mu = dy.parameter(self.params['W_mu'])
        W_sigma = dy.parameter(self.params['W_sigma'])
        R = dy.parameter(self.params['R'])
        b = dy.parameter(self.params['b'])

        batch_size = len(X_batch[0])

        if training:
            self.encoder[0].set_dropouts(self.encoder_dropout, 0)
            self.encoder[1].set_dropouts(self.encoder_dropout, 0)
            self.decoder.set_dropouts(self.decoder_dropout, 0)
        else:
            self.encoder[0].set_dropouts(0, 0)
            self.encoder[1].set_dropouts(0, 0)
            self.decoder.set_dropouts(0, 0)

        #encode
        X_ = [ dy.lookup_batch(self.params['W_emb'], tok_batch) for tok_batch in X_reverse ]
        X = [ dy.lookup_batch(self.params['W_emb'], tok_batch) for tok_batch in X_batch ]

        lstm = self.encoder[0].initial_state()
        states = lstm.transduce(X)
        forward = states[-1]

        lstm = self.encoder[1].initial_state()
        states = lstm.transduce(X_)
        backward = states[-1]

        h_t = dy.concatenate([forward, backward])

        #sample latent variable
        mu = W_mu * h_t
        sigma = dy.log(1 + dy.exp(W_sigma * h_t))     #softplus
        eps = dy.random_normal(self.latent_dim, batch_size=batch_size)
        z = mu + dy.cmult(sigma, eps)

        #decode
        lstm = self.decoder.initial_state()
        states = lstm.add_inputs([z for i in range(0, maxlen)])

        #logits
        decoding = [ dy.affine_transform([b, R, state.h()[-1]]) for state in states ]
        return decoding, mu, sigma

    def kl_divergence(self, mu, sigma):
        return 0.5 * dy.sum_elems(dy.square(mu) + dy.square(sigma) - dy.log(dy.square(sigma)) - 1)

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

        decoding, mu, sigma = self.one_sequence_batch(X_batch, X_rev, len(y_batch), X_masks, training=training)
        
        batch_loss = []
        for x, y, mask in zip(decoding, y_batch, y_masks):
            mask_expr = dy.inputVector(mask)
            mask = dy.reshape(mask_expr, (1,), batch_size)
            batch_loss.append(mask * dy.pickneglogsoftmax_batch(x, y))
        batch_loss = dy.esum(batch_loss)
        batch_loss = dy.sum_batches(batch_loss)

        divergence_loss = dy.sum_batches(self.kl_divergence(mu, sigma))

        return batch_loss, divergence_loss, decoding

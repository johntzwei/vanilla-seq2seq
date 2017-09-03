from main import neg_log_likelihood

from keras.models import Model
from keras.layers import Input
from keras.engine.topology import Layer
from keras.layers.core import Dense, Dropout, Activation, Lambda, RepeatVector, \
        Masking, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Concatenate, Add, multiply

from keras.utils import plot_model
import keras.backend as K

class CustomLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomLossLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        self.add_loss(inputs, inputs=inputs)
        return inputs

def exp_annealing(x):
    #TODO need to clip the likelihood loss
    return x[0] + K.exp(-K.stop_gradient(x[0])) * x[1]

def kl_loss(x):
    mu, sigma = x[0], x[1]
    return -0.5 * K.sum(1 + K.log(K.epsilon()+sigma) - K.square(mu) - sigma)

def vae_lm(vocab_size=10000, input_length=100, embedding_dim=64, encoder_hidden_dim=128, \
        decoder_hidden_dim=128, latent_dim=64, encoder_dropout=0.5):
    inputs = Input(shape=(input_length,))
    masked = Masking()(inputs)

    embedding_layer = Embedding(input_dim=vocab_size+1, output_dim=embedding_dim, \
            embeddings_initializer='orthogonal', input_length=input_length, mask_zero=True)
    embeddings = embedding_layer(masked)
    x = embeddings

    x1 = LSTM(encoder_hidden_dim, input_shape=(input_length, embedding_dim), unroll=True, \
            return_sequences=True)(x)
    x1 = Dropout(encoder_dropout)(x1)
    x1 = LSTM(encoder_hidden_dim, input_shape=(input_length, embedding_dim), unroll=True, \
            return_sequences=True)(x1)
    x1 = Dropout(encoder_dropout)(x1)
    x1 = LSTM(encoder_hidden_dim, input_shape=(input_length, embedding_dim), unroll=True, \
            return_sequences=True)(x1)

    x2 = LSTM(encoder_hidden_dim, input_shape=(input_length, embedding_dim), unroll=True, \
            return_sequences=True, go_backwards=True)(x)
    x2 = Dropout(encoder_dropout)(x2)
    x2 = LSTM(encoder_hidden_dim, input_shape=(input_length, embedding_dim), unroll=True, \
            return_sequences=True)(x2)
    x2 = Dropout(encoder_dropout)(x2)
    x2 = LSTM(encoder_hidden_dim, input_shape=(input_length, embedding_dim), unroll=True, \
            return_sequences=True)(x2)

    x = Add()([x1, x2])
    x = Lambda(lambda x: K.sum(x, axis=-2))(x)

    mu = Dense(latent_dim)(x)
    sigma = Dense(latent_dim, activation='softplus')(x)
    z = Lambda(lambda x: x[0] + x[1] * K.random_normal(shape=(latent_dim,), mean=0., stddev=1.))([mu, sigma])

    #sum of sentence word embeddings
    x = Lambda(lambda x: K.sum(x, axis=-1))(embeddings)
    x = Concatenate(axis=-1)([x, z])
    x = RepeatVector(input_length)(x)

    x = LSTM(decoder_hidden_dim, input_shape=(input_length, embedding_dim), unroll=True, \
            return_sequences=True)(x)

    #loss calculations
    dist_loss = Lambda(kl_loss, name='dist_loss')([mu, sigma])
    one_hot = Embedding(input_dim=vocab_size, output_dim=vocab_size, \
            embeddings_initializer='identity', mask_zero=True, trainable=False)(masked)
    xent = Lambda(lambda x: neg_log_likelihood(x[0], x[1]), output_shape=(1,), name='xent')([one_hot, x])
    loss = Lambda(exp_annealing, output_shape=(1,))([xent, dist_loss])
    x = CustomLossLayer(name='loss')(loss)

    encoder = Model(inputs=inputs, outputs=[mu, sigma])
    model = Model(inputs=inputs, outputs=[xent, dist_loss, x])
    return encoder, model

if __name__ == '__main__':
    encoder, vae = vae_lm()
    plot_model(vae, to_file='ae.png')

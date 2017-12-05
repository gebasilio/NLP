from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical

from keras.models import Input, Model
from keras.layers import Embedding, LSTM, Dense

def get_words_embedding_model(num_encoder_tokens, num_decoder_tokens, latent_dim = 256):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None,))
    x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
    x, state_h, state_c = LSTM(latent_dim,
                            return_state=True)(x)
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    decoder_embedd = Embedding(num_decoder_tokens, latent_dim)
    x = decoder_embedd(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    x,_,_ = decoder_lstm(x, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(x)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

def get_words_embedding_model_inference(num_encoder_tokens, num_decoder_tokens, latent_dim = 256):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None,))
    x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
    x, state_h, state_c = LSTM(latent_dim,
                            return_state=True)(x)
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    decoder_embedd = Embedding(num_decoder_tokens, latent_dim)
    x = decoder_embedd(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    x,_,_ = decoder_lstm(x, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(x)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(latent_dim,), name="State_input_h")
    decoder_state_input_c = Input(shape=(latent_dim,), name="State_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    x = decoder_embedd(decoder_inputs)
    decoder_outputs, state_h, state_c = decoder_lstm(x, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return model, encoder_model, decoder_model

def get_data_as_words(data_path = 'fra-eng/fra.txt', num_samples = 10000):
    tokenizer_en = Tokenizer(num_words=None, filters='"#$%&()*+,/:;=?@[\\]^_`{|}~\t\n',
               lower=True,
               split=" ",
               char_level=False)
    tokenizer_fr = Tokenizer(num_words=None, filters='',
                   lower=True,
                   split=" ",
                   char_level=False)
    
    english_sentences = []
    french_sentences = []
    
    lines = open(data_path).read().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:
        line_split = line.split('\t')
        english_sentence = [line_split[0].replace('.',' . ').replace('!',' ! ').replace('?',' ? ').replace(',',' , ').replace("'"," '")]
        english_sentences = english_sentences + english_sentence
        french_sentence = ['<START> ' 
                            + line_split[1].replace('.',' . ').replace('\u202f',' ').replace('!',' ! ').replace('?',' ? ').replace(',',' , ').replace("'"," '")
                            + '<STOP>']
        french_sentences = french_sentences + french_sentence

    tokenizer_en.fit_on_texts(english_sentences)
    tokenizer_fr.fit_on_texts(french_sentences)
    
    idx2Word_en={0:'<PAD>'}
    for key, value in tokenizer_en.word_index.items():
        idx2Word_en[value] = key
    
    idx2Word_fr={0:'<PAD>'}
    for key, value in tokenizer_fr.word_index.items():
        idx2Word_fr[value] = key
    
    
    encoder_input_data = tokenizer_en.texts_to_sequences(english_sentences)
    encoder_input_data = pad_sequences(encoder_input_data, maxlen=None, dtype='int32', padding='post', truncating='post', value=0)
    french_tokenized = tokenizer_fr.texts_to_sequences(french_sentences)
    french_tokenized = pad_sequences(french_tokenized, maxlen=None, dtype='int32', padding='post', truncating='post', value=0)
    decoder_input_data = french_tokenized[:,:-1]
    decoder_target_data = french_tokenized[:,1:]
    word2Idx_en = tokenizer_en.word_index
    word2Idx_fr = tokenizer_fr.word_index
    num_encoder_tokens = len(idx2Word_en)
    num_decoder_tokens = len(idx2Word_fr)
    decoder_target_data_cat = to_categorical(decoder_target_data.reshape(1,-1)[0]).reshape(len(decoder_target_data), decoder_target_data.shape[1], num_decoder_tokens)
    
    return english_sentences, french_sentences, \
            encoder_input_data, \
            decoder_input_data, \
            decoder_target_data_cat, \
            num_encoder_tokens, num_decoder_tokens, \
            idx2Word_en, idx2Word_fr, \
            word2Idx_en, word2Idx_fr
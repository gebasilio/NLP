from translator_helper import get_data_as_words, get_words_embedding_model
import numpy as np
from keras.callbacks import ModelCheckpoint

def main():
    num_samples = 28500
    english_sentences, french_sentences, \
    encoder_input_data, decoder_input_data, decoder_target_data, \
    num_encoder_tokens, num_decoder_tokens, \
    idx2Word_en, idx2Word_fr, \
    word2Idx_en, word2Idx_fr = get_data_as_words(num_samples = num_samples)
    latent_dim = 128
    model = get_words_embedding_model(num_encoder_tokens, num_decoder_tokens, latent_dim = latent_dim)
    model.summary()
    # Compile & run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # Note that `decoder_target_data` needs to be one-hot encoded,
    # rather than sequences of integers like `decoder_input_data`!
    epochs = 100
    batch_size = 128
    file_name = 'seq2seq_enc_dec_model_'+str(latent_dim)+'_'+str(num_samples)
    checkpoint = ModelCheckpoint(file_name+'_best.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks = callbacks_list)

    file_name = 'seq2seq_enc_dec_model_'+str(latent_dim)+'_'+str(num_samples)
    model.save_weights(file_name+'.hdf5')
    np.save(file_name, model.history.history)

if __name__ == "__main__": main()



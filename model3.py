from keras.models import Model,load_model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint

import numpy as np
import pandas as pd
import os

buffer_size = 10000
batch_size = 64
epochs = 100
latent_dim = 256

df = pd.read_csv('corpus.csv')
print(len(df),"Lines loaded")

checkpoints =  os.listdir('checkpoints')
last_data_batch = 0

for checkpoint in checkpoints:
    ckpt = checkpoint.split('_')
    print(ckpt[1])
    if (last_data_batch < int(ckpt[1])):
        last_data_batch = int(ckpt[1])

print("Starting from data batch number",last_data_batch)

for j,corpus in enumerate(np.array_split(df,92)):
    if (last_data_batch > j):
        print("Skipping data batch number",j)
        continue
    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()

    for _,row in corpus.iterrows():
        input_text = row['context']
        target_text = '\t'+row['reply']+'\n'
        
        input_texts.append(input_text)
        target_texts.append(target_text)

        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens))

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', metrics=['accuracy'], loss='categorical_crossentropy')

    checkpoint_path = "checkpoints/training_{}/cp.ckpt".format(j)
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    checkpoint = ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1)
    callbacks_list = [checkpoint]

    try:
        model.load_weights(checkpoint_path)
    except:
        pass
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks_list,
            validation_split=0.2)

    model.save('model_3.h5')
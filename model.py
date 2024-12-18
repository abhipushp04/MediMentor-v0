def build_seq2seq_model(vocab_size, embedding_dim=128, latent_dim=512):
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model, encoder_inputs, decoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense

def prepare_target_data(answer_sequences, vocab_size, max_seq_length):
    decoder_target_data = np.zeros((len(answer_sequences), max_seq_length, vocab_size), dtype='float32')
    for i, seq in enumerate(answer_sequences):
        for t, word_index in enumerate(seq):
            if t > 0:
                decoder_target_data[i, t - 1, word_index] = 1.0
    return decoder_target_data

def train_model(model, question_sequences, answer_sequences, decoder_target_data, batch_size=16, epochs=10):
    model.fit([question_sequences, answer_sequences], decoder_target_data,
              batch_size=batch_size, epochs=epochs, validation_split=0.2)

def create_inference_models(encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense, latent_dim):
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return encoder_model, decoder_model

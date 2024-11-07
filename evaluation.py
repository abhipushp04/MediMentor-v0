# Evaluating the model
def decode_sequence(input_seq, encoder_model, decoder_model, tokenizer, max_seq_length):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']

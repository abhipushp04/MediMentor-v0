# Evaluating the model
def decode_sequence(input_seq, encoder_model, decoder_model, tokenizer, max_seq_length):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index)

        if sampled_word:
            decoded_sentence += ' ' + sampled_word
            
        if (sampled_word == '<end>' or len(decoded_sentence) > max_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence


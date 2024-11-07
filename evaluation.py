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

# 10. Example usage:
filepath = '/kaggle/input/comprehensive-medical-q-a-dataset/train.csv'
questions, answers = load_and_preprocess_data(filepath)
question_sequences, answer_sequences, tokenizer, vocab_size = tokenize_and_pad_sequences(questions, answers)

plot_word_frequency(tokenizer, top_n=30)
plot_sequence_length_distribution(question_sequences, title="Question Sequence Length Distribution")
plot_sequence_length_distribution(answer_sequences, title="Answer Sequence Length Distribution")

model, encoder_inputs, decoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense = build_seq2seq_model(vocab_size)
decoder_target_data = prepare_target_data(answer_sequences, vocab_size, 150)
train_model(model, question_sequences, answer_sequences, decoder_target_data)
encoder_model, decoder_model = create_inference_models(encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense, 512)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        total_target_char_count = len(target_vocab.char2id)
        self.char_output_projection = nn.Linear(hidden_size, total_target_char_count)
        # So we don't use pre-trained embed vectors, like GLOVE in these 2 assignments. Could it be used to improve?
        self.decoderCharEmb = nn.Embedding(total_target_char_count, char_embedding_size,
                                           padding_idx=target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab
        ### END YOUR CODE
    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        embeded_input = self.decoderCharEmb(input)                          # shape (length, batch, char_embedding_size)
        output, dec_hidden = self.charDecoder(embeded_input, dec_hidden)    # output shape (length, batch, hidden_size)
        scores = self.char_output_projection(output)
        return scores, dec_hidden
        ### END YOUR CODE 

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        input = char_sequence[:-1]
        targets = char_sequence[1:]       # (length - 1, batch)
        logits, _ = self.forward(input, dec_hidden)     # (length - 1, batch, char_count)
        # Use ignore_index to filter the padded chars
        loss = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum')
        output = loss(logits.view(-1, len(self.target_vocab.char2id)), targets.contiguous().view(-1))
        return output
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        _, batch, _ = initialStates[0].size()
        target_vocab = self.target_vocab
        dec_hidden = initialStates
        softmax = nn.Softmax(dim=-1)
        output_words = ["" for _ in range(batch)]
        input = torch.tensor([target_vocab.char2id['{'] for _ in range(batch)], device=device).unsqueeze(0)    # (1, batch)
        for t in range(max_length):
            # Need a tensor whose shape is (length, batch)
            logits, dec_hidden = self.forward(input, dec_hidden)
            scores = softmax(logits)        # (1, batch, char_count)
            input = torch.argmax(scores, dim=-1)
            output_chars = input.squeeze(0).tolist()
            output_words = [(output_words[i] + target_vocab.id2char[output_chars[i]]) for i in range(batch)]
        decoded_words = [output_word[:output_word.find("}")] if "}" in output_word else output_word for output_word in output_words]
        return decoded_words
        ### END YOUR CODE


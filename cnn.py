#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, char_embed_size, word_embed_size, max_word_length=21, kernel_size=5):
        # Use a kernel size k=5
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(char_embed_size, word_embed_size, kernel_size)
        self.max_pool = nn.MaxPool1d(max_word_length - kernel_size + 1)

    def forward(self, x_reshaped):
        """
        :param x_reshaped: The tensor of char embedding before into the conv network. Size: [batch_size, char_embed_size, max_word_length]
        :return: x_conv_out: The tensor out of conv layer. Size: [batch_size, word_embed_size]
        """
        x_conv = self.conv1d(x_reshaped)        # [batch_size, word_embed_size, (max_word_length - kernel_size + 1)]
        x_conv_out = self.max_pool(x_conv).squeeze(-1)      # [batch_size, word_embed_size]
        return x_conv_out

### END YOUR CODE


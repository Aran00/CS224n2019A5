#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, word_embed_size):
        """
        :param word_embed_size: The embed size of word
        """
        super(Highway, self).__init__()
        self.proj = nn.Linear(word_embed_size, word_embed_size)
        self.gate = nn.Linear(word_embed_size, word_embed_size)

    def forward(self, x_conv_out):
        """
        :param x_conv_out: The tensor out of conv layer [batch_size, word_embed_size]
        :return x_highway: The tensor out of highway network [batch_size, word_embed_size]
        """
        # Operate on batches of words
        # Be sure to use 2 nn.Linear layers
        x_proj = F.relu(self.proj(x_conv_out))
        x_gate = torch.sigmoid(self.proj(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        return x_highway


### END YOUR CODE

'''
### No sanity test, you will now write your own code to thoroughly test your implementation
• Write code to check that the input and output have the expected shapes and types. Before you
do this, make sure you've written docstrings for init () and forward() { you can't test
the expected output if you haven't clearly laid out what you expect it to be!
• Print out the shape of every intermediate value; verify all the shapes are correct.
• Create a small instance of your highway network (with small, manageable dimensions), manually
define some input, manually dene the weights and biases, manually calculate what the output
should be, and verify that your module does indeed return that value.
• Similar to previous, but you could instead print all intermediate values and check each is correct.
• If you can think of any `edge case' or `unusual' inputs, create test cases based on those.
'''



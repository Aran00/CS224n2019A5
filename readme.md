### The Points in A4

#### run.py
1. The calculation of PPL：Average to each word
2. When evaluating dev set ppl during training process, we need to call model.eval() first. After evaluation, call model.train() again.
3. The common training process in pytorch again:
    ```bash
    model = ...
    optimizer = ...
    while epoch < max_epoch_limit:
        foreach batch:
            optimizer.zero_grad()
            loss = f(model, feature_data, target_data)     # A tensor
            # All kinds of logging
            loss.backward()
            # clip norm if needed here using clip_grad_norm_
            optimizer.step()
    ```
4. The other things needed to note in the training process:
    
    a. We need to copy both the model and data to device(In this file, copying data to device is done by the model class)

    b. Grad clipping
    
    c. Log. What should be printed to the output?
    
    d. Patience settings: If the validation ppl decrease times hits the patience then
       
      - Restore model from the optimal checkpoint, and copy it to the device           
      - Decrease the learning rate, and reset the patience to 0, then try again
      - If the above step trails reaches the max trail number, early stop
    
    e. The condition of early stop(contained above)

5. Is with torch.no_grad() necessary in module evaluation?

#### model_embeddings.py
1. We don't use pre-trained embedding word vectors, like GLOVE in this assignment.
So the word embedding is also a param that needs to be trained. 
Could try to use context-based vectors (like ELMo) and see if it can improve the results.
2. When init nn.Embedding(), need to note to give padding_idx(I think the padding vector would not be trained)

#### nmt_model.py
1. The model construction is the central point. How to test?
   
   a. UT. Check the input/output by some fixed data
   
   b. tensor size check. Also could be contained in #a
   
   c. Train the model quickly on a mini-batch and see its overfit. 

2. Really need a good software to draw the model structure...

3. Maybe we could let the params of forward() function to be tensor? I think the model shouldn't concern its device...

4. The unique problems of NLP model: padding/mask. First, we need to padding source and target data. 
While in Seq2Seq model construction, how should it be used?
    
    a. The target padding/mask should be used when calculating the final loss. 
    Of course, the masked(padded) position shouldn't have loss.
    
    b. The usage of padding/mask:
       
      - For pyTorch, source_length is needed for packing.
      - Before input into decoder, need to produce a mask according to the source length.
      (Used in attention calculation below) In the decode process, when implementing the attention, 
      we also need to remove the padded position. The method is using mask to set the padded position in the logit to -inf.
      Then after softmax, the weight of these padded positions would be 0. 
      - The target mask is used when calculating the final loss. The padded position would not be contained in the final loss.  
    So the convention is to make the padding position to 1 in the mask?
    Try to summarize their usages later.

5.  The general workflow in a torch module:

    a. init layers in init function
    
    b. call these layers one by one in forward function
    
    c. return some tensor at last. It could be a loss or the final output tensor, 
    depending on whether you want to assign the loss calculation method in the 
    main workflow.(I think moving it outside would be better. However, in a NLP model,
    calculating final loss needs to use the target mask, maybe the author 
    does not want to expose these details outside.)
    
6. In the forward function, the whole workflow could be divided to different sub-modules
    (like encoder and decoder in this example). The critical thing here is to make clear what's
    the inputs and outputs (also size) of each sub-module.
    
7. For RNN, if we have paddings in source data, using packing is necessary(must in bidirectional!),
   or the padded position would be included in the forward computation.
   It contains 2 functions:
   ```
   pack_padded_sequence   # Before RNN input
   pad_packed_sequence    # After RNN output
   ``` 
   For different value of batch_first in pack_padded_sequence, the output shape(batch dim position) is different.
   Use enforce_sorted to set whether to sort the batch by length descent or not. For the length descent case(default case),
   the input data also needs to be sorted by length desc before passing the RNN.
   
8. Remember to call contiguous() before view() (Almost always need, but remember when needs to call is also helpful)

9. In decode process, when combining the output of the last time step with the input of current, we can only use a RNN cell to forward
   but not a whole RNN.
  
10. In seq2sql decode process, when feeding the target to the input, it needs to be truncated. 
   The last time item <END> would be chopped while the first <START> will not(The reason is obvious)
     
11. In decoding process, when implementing the attention, use batch Multiplication function torch.bmm() to do a 3D matrix multiply to get logit for each batch item(For the simplest self attention).
   The final vector used for multiply the weights needn't be the vectors dot multiplied with the original output vector.  
   
12. A linear module could use a batch matrix as input, and the output would change the last dim.

13. When using the seq2seq, remember to truncate the first<START> or last<END> is very important. The cases include:
    Summary it later
    
16. Beam search:
    
    a. 
    
    b. 
    
    c. In the beam search process, the beam size would decrease when some hypotheses is completed.
    Why not keep these finished hypothesis in and compare them with others? 
    
    - If their sentence continue to grow, it would be wrong
    
    - If they stop to grow, then the other hypothesis's probability would be smaller and smaller 
    after new time steps, so the finished would always be in the largest top K. So no need to compare again.
    
    - Would this cause the model to prefer shorter output sentences?
    
    d. The beam search program seems to be the most complex part in the whole program. The general workflow is as below:
      
      - Feed data into encoder to get src_encodings, and decoding initial states    
      
      - The decode process. The difficult point is that we need to select the input of decode cell manually in each timestamp.
        ```
        while t < max_decoding_time_step
            t = t + 1
            Expand src_encodings and exp_src_encodings_att_linear to the living hypothesis count batch
            Prepare cell input, including the (last predicted word embedding) concat (last hidden state attention result)
            step() to pass one LSTM cell
            flatten the softmax of LSTM cell output result, add it with the original score
            the result would be an indicator of the probability of the whole sequence. Get topk of them
            for each hypothesis in topk result
                if the hypothesis is completed
                    add it to the completed hypotheses set
                else
                    construct it into the new start hypotheses, save their sequence and score, and related hypotheses id. 
            if the completed hyp count == beam size
                break
            
            prepare the input state of the next time step by related hypotheses ids collected above.
        
        if len(completed_hypotheses) == 0:
            Only select the top 1 hypothesis
        
        Sort completed_hypotheses and return the one with the highest score
    
        ``` 
          
### The additional contents in A5

#### cnn.py

1. Add a CNN 

2. Conv1d: input channel -> char_embed_size, output channel: word_embed_size
   The kernel size is only a number as the other dim of the 1d kernel is fixed - input channel size
   The input is still a 3D tensor whose size is [batch_size, char_embed_size, max_word_length]
   and the output size is [batch_size, word_embed_size, max_word_length - kernel_size + 1]
   So we can see that the 1D conv also likes 2D conv, 
   the only difference is that it would only do conv calculations on the word length dim. 
   The whole embed size dim would totally be contained in the conv kernel area. 
   
   Compare to 2D conv:
   [batch_size, input_channel, height, width] -> [batch_size, output_channel, kernelled_height, kernelled_width]
   We can see they have the similar pattern.

#### model_embeddings.py

Add CNN - highway layer.
1. The batch_size & sentence length dims should be flattened, as in char level CNN, the CNNed target is a word.
And in the final step, it should be reshaped to the original size.

#### char_decoder.py


#### nmt_model.py
1. The first significant diff: The tensor before the embedding layer is char vector now

2. The role of the char decoder:
   a. 
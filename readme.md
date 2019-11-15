### The Points

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
    
14. Beam search:
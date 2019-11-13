### The Points

#### run.py
1. The calculation of PPL：Average to each word
2. When evaluating dev set ppl during training process, we need to call model.eval() first. After evaluation, call model.train() again.
3. The common training process in pytorch again:
    ```bash
    optimizer.zero_grad()
    loss = ...     # A tensor
    loss.backward()
    optimizer.step()
    ```
4. The other things needed to note in the training process:
    
    a. We need to copy both the model and data to device

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

### Why my network not works properly?

lets make some changes to our codebase to make it work.

### What I did test?

- `[NO EFFECT]` Use character based tokenizer to both encoder and decoder. so give up the subword level tokenizer from decoder.
- `[NO EFFECT]` Add layer normalization layer to our transformer network.
- `[NO EFFECT]` use same embedding layer to both encoder and decoder layer
- `[NO EFFECT]` use float32 instead of float16.
- `[NO EFFECT]` modify memory key padding mask to use the same key padding mask for encoder and decoder.
- `[NO EFFECT]` remove autoregressive training.
- `[NO EFFECT]` shirnk model parameters.
- `[NO EFFECT]` ask cursor to implement generate_memory, forward, generate_text function.
- `[NO TESTED]` Swap models to test model and training loop.
- `[NO EFFECT]` add math.sqrt(d_model) after embedding in forward method on model.

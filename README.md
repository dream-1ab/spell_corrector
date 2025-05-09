# Spell correction transformer based neural network model for Uyghur language.

# What?
spell checking and correction is the one most NLP tasks we needed in our projects. from simple keyboard app to text dataset pre-processing, auto speech recognition, web/database search, language modeling, text normalization and more.

this is an open source spell correction language model based on transformer based neural network. we believe we are the first developer releases such language model.

we train this model with text corpus scraped from the internet and provide pre-trained model. you can fine-tune this model on your specific tasks by follow our train/fine-tune documentation.


# Why:
here are some use case of this language model but not limited:
- Keyboard/Input method app. correct user typings.
- ASR (Automatic speech recognition). ASR models may perform much much worse without spell correction.
- Standlone spell correction applications.
- Dataset pre-processing tasks for other language modeling. e.g. Large language models. machine translation models even TTS Models.
- Search (web search or database search). we cannot search something in our database unless the key of data we want to search is match 1:1

# Who can use this language model?

Anyone who needed. whatever open source or closed source, free or commericial, personal or organization anyone can use this model.

# Architecture
we used the standard Encoder/Decoder transformer architecture from pytorch machine learning library.

- Encoder: we use character level tokenization for encoder layer to avoid oov (out of vocabulary) issue.
- Decoder: BPE (Byte pair encoding) tokenization to decoder layer for performance and fast training. we use decoder model as autoregressive model for better output quality

# TODO
- [TODO] Pre-train our model.
- [TODO] Publish our pre-trained model on HuggingFace.
- [TODO] make instructions to Train/Fine-tune on custom dataset or train from scratch.
- [TODO] Provide ONNX exported model file.
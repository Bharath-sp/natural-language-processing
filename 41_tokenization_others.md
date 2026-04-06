---
layout: post
title: Other Topics in Tokenization
categories: [NLP]
toc: true
---

* TOC
{:toc}


## ByteT5
Towards a Token-Free Future with Pre-trained Byte-to-Byte Models

ByteT5 is a token-free method. For any task in NLP, we give input to the model, hope that the model can understand it, and processes it accordingly to give us the output. Nowhere it is mentioned to tokenize the text. In general, the text data is generally stored as a sequence of bytes. Thus, feeding byte sequences directly into the model (such as a sequence-to-sequence model) enables the processing of arbitrary text sequences.

UTF-8 bytes are fed directly into the model without any text preprocessing.

It is found out that the ByT5 performs the best when the depth of the encoder and decoder stacks are decoupled. It benefits significantly from a heavier encoder i.e., number of encoder layers to be 3 times more than that of the number of decoder layers.



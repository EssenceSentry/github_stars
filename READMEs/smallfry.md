[![Build Status](https://travis-ci.com/HazyResearch/smallfry.svg?branch=master)](https://travis-ci.com/HazyResearch/smallfry)

# SmallFry
Word embeddings are a key component of modern NLP models. To attain strong performance on various
tasks, it is often necessary to use a very large vocabulary, and/or high-dimensional embeddings.
As a result, word embeddings can consume a large amount of memory during training and inference. 

**Smallfry is a simple word embedding compression algorithm based on uniform quantization.** It first automatically clips the extreme values of a pre-trained embedding matrix, and then compresses the clipped embeddings using uniform quantization. Once the embeddings are compressed, they can be used to significantly lower the memory for training or inference for NLP models.

Our PyTorch QuantEmbedding module can be used as a drop-in replacement for the PyTorch Embedding module.

For more information about our compression algorithm, along with corresponding theoretical analysis, please see our [NeurIPS 2019 paper](https://arxiv.org/pdf/1909.01264.pdf).

## Installation
To install the smallfry package, please clone and pip install our repository as follows:
```
git clone --recursive https://github.com/HazyResearch/smallfry.git
cd smallfry
pip install -e .
```
Our implementation is tested under Python 3.6 and PyTorch 1.0.

## Usage
### Directly initialize a compressed embedding layer
The parameters for initializing a QuantEmbedding module are the same as those for the [PyTorch Embedding module](https://pytorch.org/docs/stable/nn.html#embedding). The only additional required parameter is ```nbit```, which specifies the number of bits to use per value of the compressed embedding matrix. Currently we support 1, 2, 4, 8, 16, and 32 bit representations. During initialization, the pre-trained embedding values can be loaded via a ```torch.FloatTensor``` or via a file in GloVe format (no file header), where every line represents a word vector. Below, we show examples of both of these initialization strategies for the QuantEmbedding module:

```
from smallfry import QuantEmbedding

# init with existing tensor
embed_from_tensor = 
    QuantEmbedding(num_embeddings=1000,    # vocabulary size
                   embedding_dim=50,       # embedding dimensionality
                   _weight=<a PyTorch FloatTensor (rows are embeddings),
                   nbit=4)                 # the quantization precision

# init with embedding files
embed_from_file = 
    QuantEmbedding(num_embeddings=1000,    # vocabulary size
                   embedding_dim=50,       # embedding dimensionality
                   nbit=2,                 # the quantization precision
                   embedding_file=<a GloVe format embedding file>)
```
If the input embedding matrix is uncompressed, the QuantEmbedding module will automatically compress it to the specified number of bits per entry. If the embedding matrix is already compressed (meaning its number of unique values is equal to 2^n_bit), the QuantEmbedding module will directly use these values without performing any additional compression.

### Replace an existing embedding layer with a quantized embedding layer
Given an existing model with one or more Embedding modules, one may want to replace all these modules with QuantEmbedding modules.  This can be done using the following helper function which we provide:

```
from smallfry import quantize_embed
...
quantize_embed(model,   # a model, i.e. an instance of PyTorch nn.Module,
               nbit=2,  # the quantization precision)
```

## End-to-end example
We present an end-to-end example for how to use the QuantEmbedding module for training a question-answering system using less memory. In this example, we train a LSTM-based [DrQA](https://github.com/facebookresearch/DrQA) model on the [SQuAD1.1](https://rajpurkar.github.io/SQuAD-explorer/) dataset. We train the DrQA model on top of a fixed pre-trained [GloVe](https://nlp.stanford.edu/projects/glove/) embedding, using 2-bit quantization.

### Setup
We provide the following script to automatically download the required data and install the DrQA package.
```
bash prepare_drqa.sh
```

### Run
After the setup is done, the training using compressed embeddings can be launched via
```
bash run_drqa.sh
```
When training completes, the 2-bit compressed embeddings attain a F1 score of ~73.72% on the dev set, whereas the uncompressed embeddings attain ~73.86% dev set F1 score.

# What is Gensim-data for?

Research datasets regularly disappear, change over time, become obsolete or come without a sane implementation to handle the data format reading and processing.

For this reason, [Gensim](https://github.com/RaRe-Technologies/gensim) launched its own dataset storage, committed to long-term support, a sane standardized usage API and focused on datasets for **unstructured text processing** (no images or audio). This [Gensim-data](https://github.com/RaRe-Technologies/gensim-data) repository serves as that storage.

**There's no need for you to use this repository directly**. Instead, simply install Gensim and use its download API (see the Quickstart below). It will "talk" to this repository automagically.

💡 When you use the Gensim download API, all data is stored in your `~/gensim-data` home folder.

Read more about the project rationale and design decisions in this article: [New Download API for Pretrained NLP Models and Datasets](https://rare-technologies.com/new-download-api-for-pretrained-nlp-models-and-datasets-in-gensim/).

# How does it work?

Technically, the actual (sometimes large) corpora and model files are being stored as [release attachments](https://github.com/RaRe-Technologies/gensim-data/releases) here on Github. Each dataset (and each new version of each dataset) gets its own release, forever immutable.

Each release is accompanied by a usage example and release notes, for example: [Corpus of USPTO Patents from 2017](https://github.com/RaRe-Technologies/gensim-data/releases/tag/patent-2017); [English Wikipedia from 2017 with plaintext section](https://github.com/RaRe-Technologies/gensim-data/releases/tag/wiki-english-20171001).

🔴 **Each dataset comes with its own license, which the users should study carefully before using the dataset!**

----

## Quickstart

To load a model or corpus, use either the Python or command line interface of [Gensim](https://github.com/RaRe-Technologies/gensim) (you'll need Gensim installed first):

- **Python API**

  Example: load a pre-trained model (gloVe word vectors):

  ```python
  import gensim.downloader as api

  info = api.info()  # show info about available models/datasets
  model = api.load("glove-twitter-25")  # download the model and return as object ready for use
  model.most_similar("cat")

  """
  output:

  [(u'dog', 0.9590819478034973),
   (u'monkey', 0.9203578233718872),
   (u'bear', 0.9143137335777283),
   (u'pet', 0.9108031392097473),
   (u'girl', 0.8880630135536194),
   (u'horse', 0.8872727155685425),
   (u'kitty', 0.8870542049407959),
   (u'puppy', 0.886769711971283),
   (u'hot', 0.8865255117416382),
   (u'lady', 0.8845518827438354)]

  """
  ```

  Example: load a corpus and use it to train a Word2Vec model:

  ```python
  from gensim.models.word2vec import Word2Vec
  import gensim.downloader as api

  corpus = api.load('text8')  # download the corpus and return it opened as an iterable
  model = Word2Vec(corpus)  # train a model from the corpus
  model.most_similar("car")

  """
  output:

  [(u'driver', 0.8273754119873047),
   (u'motorcycle', 0.769528865814209),
   (u'cars', 0.7356342077255249),
   (u'truck', 0.7331641912460327),
   (u'taxi', 0.718338131904602),
   (u'vehicle', 0.7177008390426636),
   (u'racing', 0.6697118878364563),
   (u'automobile', 0.6657308340072632),
   (u'passenger', 0.6377975344657898),
   (u'glider', 0.6374964714050293)]

  """
  ```

  Example: **only** download a dataset and return the local file path (no opening):

  ```python
  import gensim.downloader as api

  print(api.load("20-newsgroups", return_path=True))  # output: /home/user/gensim-data/20-newsgroups/20-newsgroups.gz
  print(api.load("glove-twitter-25", return_path=True))  # output: /home/user/gensim-data/glove-twitter-25/glove-twitter-25.gz
  ```

 - The same operations, but from **CLI, command line interface**:

   ```bash
   python -m gensim.downloader --info  # show info about available models/datasets
   python -m gensim.downloader --download text8  # download text8 dataset to ~/gensim-data/text8
   python -m gensim.downloader --download glove-twitter-25  # download model to ~/gensim-data/glove-twitter-50/
   ```

----

## Available data

### Datasets

| name | file size | read_more | description | license |
|------|-----------|-----------|-------------|---------|
| 20-newsgroups | 13 MB | <ul><li>http://qwone.com/~jason/20Newsgroups/</li></ul> | The notorious collection of approximately 20,000 newsgroup posts, partitioned (nearly) evenly across 20 different newsgroups. | not found |
| fake-news | 19 MB | <ul><li>https://www.kaggle.com/mrisdal/fake-news</li></ul> | News dataset, contains text and metadata from 244 websites and represents 12,999 posts in total from a specific window of 30 days. The data was pulled using the webhose.io API, and because it's coming from their crawler, not all websites identified by their BS Detector are present in this dataset. Data sources that were missing a label were simply assigned a label of 'bs'. There are (ostensibly) no genuine, reliable, or trustworthy news sources represented in this dataset (so far), so don't trust anything you read. | https://creativecommons.org/publicdomain/zero/1.0/ |
| patent-2017 | 2944 MB | <ul><li>http://patents.reedtech.com/pgrbft.php</li></ul> | Patent Grant Full Text. Contains the full text including tables, sequence data and 'in-line' mathematical expressions of each patent grant issued in 2017. | not found |
| quora-duplicate-questions | 20 MB | <ul><li>https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs</li></ul> | Over 400,000 lines of potential question duplicate pairs. Each line contains IDs for each question in the pair, the full text for each question, and a binary value that indicates whether the line contains a duplicate pair or not. | probably https://www.quora.com/about/tos |
| semeval-2016-2017-task3-subtaskA-unannotated | 223 MB | <ul><li>http://alt.qcri.org/semeval2016/task3/</li> <li>http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016-task3-report.pdf</li> <li>https://github.com/RaRe-Technologies/gensim-data/issues/18</li> <li>https://github.com/Witiko/semeval-2016_2017-task3-subtaskA-unannotated-english</li></ul> | SemEval 2016 / 2017 Task 3 Subtask A unannotated dataset contains 189,941 questions and 1,894,456 comments in English collected from the Community Question Answering (CQA) web forum of Qatar Living. These can be used as a corpus for language modelling. | These datasets are free for general research use. |
| semeval-2016-2017-task3-subtaskBC | 6 MB | <ul><li>http://alt.qcri.org/semeval2017/task3/</li> <li>http://alt.qcri.org/semeval2017/task3/data/uploads/semeval2017-task3.pdf</li> <li>https://github.com/RaRe-Technologies/gensim-data/issues/18</li> <li>https://github.com/Witiko/semeval-2016_2017-task3-subtaskB-english</li></ul> | SemEval 2016 / 2017 Task 3 Subtask B and C datasets contain train+development (317 original questions, 3,169 related questions, and 31,690 comments), and test datasets in English. The description of the tasks and the collected data is given in sections 3 and 4.1 of the task paper http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016-task3-report.pdf linked in section “Papers” of https://github.com/RaRe-Technologies/gensim-data/issues/18. | All files released for the task are free for general research use |
| text8 | 31 MB | <ul><li>http://mattmahoney.net/dc/textdata.html</li></ul> | First 100,000,000 bytes of plain text from Wikipedia. Used for testing purposes; see wiki-english-* for proper full Wikipedia datasets. | not found |
| wiki-english-20171001 | 6214 MB | <ul><li>https://dumps.wikimedia.org/enwiki/20171001/</li></ul> | Extracted Wikipedia dump from October 2017. Produced by `python -m gensim.scripts.segment_wiki -f enwiki-20171001-pages-articles.xml.bz2 -o wiki-en.gz` | https://dumps.wikimedia.org/legal.html |

### Models

| name | num vectors | file size | base dataset | read_more  | description | parameters | preprocessing | license |
|------|-------------|-----------|--------------|------------|-------------|------------|---------------|---------|
| conceptnet-numberbatch-17-06-300 | 1917247 | 1168 MB | ConceptNet, word2vec, GloVe, and OpenSubtitles 2016 | <ul><li>http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14972</li> <li>https://github.com/commonsense/conceptnet-numberbatch</li> <li>http://conceptnet.io/</li></ul> | ConceptNet Numberbatch consists of state-of-the-art semantic vectors (also known as word embeddings) that can be used directly as a representation of word meanings or as a starting point for further machine learning. ConceptNet Numberbatch is part of the ConceptNet open data project. ConceptNet provides lots of ways to compute with word meanings, one of which is word embeddings. ConceptNet Numberbatch is a snapshot of just the word embeddings. It is built using an ensemble that combines data from ConceptNet, word2vec, GloVe, and OpenSubtitles 2016, using a variation on retrofitting. | <ul><li>dimension - 300</li></ul> | - | https://github.com/commonsense/conceptnet-numberbatch/blob/master/LICENSE.txt |
| fasttext-wiki-news-subwords-300 | 999999 | 958 MB | Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens) | <ul><li>https://fasttext.cc/docs/en/english-vectors.html</li> <li>https://arxiv.org/abs/1712.09405</li> <li>https://arxiv.org/abs/1607.01759</li></ul> | 1 million word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens). | <ul><li>dimension - 300</li></ul> | - | https://creativecommons.org/licenses/by-sa/3.0/ |
| glove-twitter-100 | 1193514 | 387 MB | Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased) | <ul><li>https://nlp.stanford.edu/projects/glove/</li> <li>https://nlp.stanford.edu/pubs/glove.pdf</li></ul> | Pre-trained vectors based on  2B tweets, 27B tokens, 1.2M vocab, uncased (https://nlp.stanford.edu/projects/glove/) | <ul><li>dimension - 100</li></ul> | Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-twitter-100.txt`. | http://opendatacommons.org/licenses/pddl/ |
| glove-twitter-200 | 1193514 | 758 MB | Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased) | <ul><li>https://nlp.stanford.edu/projects/glove/</li> <li>https://nlp.stanford.edu/pubs/glove.pdf</li></ul> | Pre-trained vectors based on 2B tweets, 27B tokens, 1.2M vocab, uncased (https://nlp.stanford.edu/projects/glove/). | <ul><li>dimension - 200</li></ul> | Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-twitter-200.txt`. | http://opendatacommons.org/licenses/pddl/ |
| glove-twitter-25 | 1193514 | 104 MB | Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased) | <ul><li>https://nlp.stanford.edu/projects/glove/</li> <li>https://nlp.stanford.edu/pubs/glove.pdf</li></ul> | Pre-trained vectors based on 2B tweets, 27B tokens, 1.2M vocab, uncased (https://nlp.stanford.edu/projects/glove/). | <ul><li>dimension - 25</li></ul> | Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-twitter-25.txt`. | http://opendatacommons.org/licenses/pddl/ |
| glove-twitter-50 | 1193514 | 199 MB | Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased) | <ul><li>https://nlp.stanford.edu/projects/glove/</li> <li>https://nlp.stanford.edu/pubs/glove.pdf</li></ul> | Pre-trained vectors based on 2B tweets, 27B tokens, 1.2M vocab, uncased (https://nlp.stanford.edu/projects/glove/) | <ul><li>dimension - 50</li></ul> | Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-twitter-50.txt`. | http://opendatacommons.org/licenses/pddl/ |
| glove-wiki-gigaword-100 | 400000 | 128 MB | Wikipedia 2014 + Gigaword 5 (6B tokens, uncased) | <ul><li>https://nlp.stanford.edu/projects/glove/</li> <li>https://nlp.stanford.edu/pubs/glove.pdf</li></ul> | Pre-trained vectors based on Wikipedia 2014 + Gigaword 5.6B tokens, 400K vocab, uncased (https://nlp.stanford.edu/projects/glove/). | <ul><li>dimension - 100</li></ul> | Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-wiki-gigaword-100.txt`. | http://opendatacommons.org/licenses/pddl/ |
| glove-wiki-gigaword-200 | 400000 | 252 MB | Wikipedia 2014 + Gigaword 5 (6B tokens, uncased) | <ul><li>https://nlp.stanford.edu/projects/glove/</li> <li>https://nlp.stanford.edu/pubs/glove.pdf</li></ul> | Pre-trained vectors based on Wikipedia 2014 + Gigaword, 5.6B tokens, 400K vocab, uncased (https://nlp.stanford.edu/projects/glove/). | <ul><li>dimension - 200</li></ul> | Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-wiki-gigaword-200.txt`. | http://opendatacommons.org/licenses/pddl/ |
| glove-wiki-gigaword-300 | 400000 | 376 MB | Wikipedia 2014 + Gigaword 5 (6B tokens, uncased) | <ul><li>https://nlp.stanford.edu/projects/glove/</li> <li>https://nlp.stanford.edu/pubs/glove.pdf</li></ul> | Pre-trained vectors based on Wikipedia 2014 + Gigaword, 5.6B tokens, 400K vocab, uncased (https://nlp.stanford.edu/projects/glove/). | <ul><li>dimension - 300</li></ul> | Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-wiki-gigaword-300.txt`. | http://opendatacommons.org/licenses/pddl/ |
| glove-wiki-gigaword-50 | 400000 | 65 MB | Wikipedia 2014 + Gigaword 5 (6B tokens, uncased) | <ul><li>https://nlp.stanford.edu/projects/glove/</li> <li>https://nlp.stanford.edu/pubs/glove.pdf</li></ul> | Pre-trained vectors based on Wikipedia 2014 + Gigaword, 5.6B tokens, 400K vocab, uncased (https://nlp.stanford.edu/projects/glove/). | <ul><li>dimension - 50</li></ul> | Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-wiki-gigaword-50.txt`. | http://opendatacommons.org/licenses/pddl/ |
| word2vec-google-news-300 | 3000000 | 1662 MB | Google News (about 100 billion words) | <ul><li>https://code.google.com/archive/p/word2vec/</li> <li>https://arxiv.org/abs/1301.3781</li> <li>https://arxiv.org/abs/1310.4546</li> <li>https://www.microsoft.com/en-us/research/publication/linguistic-regularities-in-continuous-space-word-representations/?from=http%3A%2F%2Fresearch.microsoft.com%2Fpubs%2F189726%2Frvecs.pdf</li></ul> | Pre-trained vectors trained on a part of the Google News dataset (about 100 billion words). The model contains 300-dimensional vectors for 3 million words and phrases. The phrases were obtained using a simple data-driven approach described in 'Distributed Representations of Words and Phrases and their Compositionality' (https://code.google.com/archive/p/word2vec/). | <ul><li>dimension - 300</li></ul> | - | not found |
| word2vec-ruscorpora-300 | 184973 | 198 MB | Russian National Corpus (about 250M words) | <ul><li>https://www.academia.edu/24306935/WebVectors_a_Toolkit_for_Building_Web_Interfaces_for_Vector_Semantic_Models</li> <li>http://rusvectores.org/en/</li> <li>https://github.com/RaRe-Technologies/gensim-data/issues/3</li></ul> | Word2vec Continuous Skipgram vectors trained on full Russian National Corpus (about 250M words). The model contains 185K words. | <ul><li>dimension - 300</li> <li>window_size - 10</li></ul> | The corpus was lemmatized and tagged with Universal PoS | https://creativecommons.org/licenses/by/4.0/deed.en |

(generated by [generate_table.py](https://github.com/RaRe-Technologies/gensim-data/blob/master/generate_table.py) based on [list.json](https://github.com/RaRe-Technologies/gensim-data/blob/master/list.json))


----

# Want to add a new corpus or model?

1. Compress your data set using gzip or bz2.

2. Share the compressed file on any file-sharing service.

2. Create a [new issue](https://github.com/RaRe-Technologies/gensim-data/issues) and give us the dataset link. Add a **detailed description** on **why** and **how** you created the dataset, any related papers or research, plus how do you expect other users should use it. Include a code example where relevant.

----------------

`Gensim-data` is open source software released under the [LGPL 2.1 license](https://github.com/rare-technologies/gensim-data/blob/master/LICENSE).

Copyright (c) 2018 [RARE Technologies](https://rare-technologies.com/).

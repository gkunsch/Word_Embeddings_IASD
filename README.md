WE_HGA

In this repository, we perform word by word translation using off-the-shelf embedding and various mapping (linear, linear with constraints, unsupervised) to perform translation. 

We rely on embeddings by Fasttext, they can be found in the embedding folder. 
We have 2 types on embeddings:
- binary format, that can later be used to find the embedding of a different word (even a word that doesn't exist)
- 'vec' format, where each row is the word + it's embedding obtained using Fasttext.

For Jupyter notebook, the direction of the translation is precised (french to english, or english to french). Besides, the number in parenthesis indicates the dimension of the embedding (limited at 300 for the embedding choosen). If the notebook's name contains 'fixed', it means the 'vec' embedding format have been used. They provide better results but don't allow for flexibility in words.
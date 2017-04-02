import os
import sys
sys.path.append(".")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import normalize
from pyemd import emd
import pickle

#inspired from http://nbviewer.jupyter.org/github/vene/vene.github.io/blob/pelican/content/blog/word-movers-distance-in-python.ipynb

if not os.path.exists("data/word_mover_data/embed.dat"):
    print("Caching word embeddings in memmapped format...")
    from gensim.models.word2vec import Word2Vec

    wv = Word2Vec.load_word2vec_format(
        "data/word_mover_data/GoogleNews-vectors-negative300.bin.gz",
        binary=True)
    wv.init_sims(replace=True)
    fp = np.memmap("data/word_mover_data/embed.dat", dtype=np.double, mode='w+', shape=wv.syn0norm.shape)
    fp[:] = wv.syn0norm[:]
    with open("data/word_mover_data/embed.vocab", "w", encoding='utf-8') as f:
        for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
            print(w, file=f)
            del fp, wv

W = np.memmap("data/word_mover_data/embed.dat", dtype=np.double, mode="r", shape=(3000000, 300))
with open("data/word_mover_data/embed.vocab", "r",encoding='utf-8') as f:
    vocab_list = map(str.strip, f.readlines())

vocab_dict = {w: k for k, w in enumerate(vocab_list)}




# docs = ["Obama speaks to the media in Illinois",
#         "The President addresses the press in Chicago",
#         "The soccer team has done a great job winning the series",
#         "It was an awesome team by the soccer team in the world series"
#         "I am a data scientist working with Deloitte",
#         "This may or may not be a relevant sentence",
#         "This is definitely not a relevant sentence",
#         "This is definitely not sentence",
#         "This is a sentence"
#         ]
#
# docs = [
# "What is the step by step guide to invest in share market in india?",
# "What is the step by step guide to invest in share market?",
# "Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?",
# "I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me?"
# ]






df = pd.read_csv('data/train.csv')

print("\nFound Nans in the following rows : ")
for i in pd.isnull(df).any(1).nonzero()[0]:
    print("\n")
    print(df.iloc[[i]])
print("\n\nRemoving Nans from the dataset")

df.dropna(inplace=True)


df = df[['question1', 'question2', 'is_duplicate']]


docs = []
input_tuples = []
for index, row in df.iterrows():
    docs.append(row['question1'])
    docs.append(row['question2'])



# docs = [
#     "How is the new Harry Potter book 'Harry Potter and the Cursed Child'?",
#     "How bad is the new book by J.K Rowling?",
#     "my name is Harry Potter",
#     "my name is J.K Rowling"
# ]

ref_dict = {
    0: "Dist_with_stopword_removal",
    1: "Dist_without_stopword_removal"
}
itr = 0


vect_with_stopword_removal = CountVectorizer(stop_words="english").fit(docs)
vect_without_stopword_removal = CountVectorizer().fit(docs)

dist_dict = dict()
dist_dict[0] = []
dist_dict[1] = []


for vect in [vect_with_stopword_removal, vect_without_stopword_removal]:

    # vect = CountVectorizer().fit(docs)
    common = [word for word in vect.get_feature_names() if word in vocab_dict]
    W_common = W[[vocab_dict[w] for w in common]]

    vect = CountVectorizer(vocabulary=common, dtype=np.double)

    X = vect.fit_transform(docs)
    X = normalize(X, norm='l1', copy=False)

    i = 0
    while i < len(docs)-1:
        union_idx = np.union1d(X[i+1].indices, X[i].indices)

        W_minimal = W_common[union_idx]
        try:
            W_dist = euclidean_distances(W_minimal)
            bow_i = X[i+1, union_idx].A.ravel()
            bow_j = X[i, union_idx].A.ravel()
            dist = emd(bow_i, bow_j, W_dist)
            dist_dict[itr].append(dist)
            i += 2
        except:
            dist_dict[itr].append(-1)
            i += 2

    se = pd.Series(dist_dict[itr])
    df[ref_dict[itr]] = se.values
    itr += 1

print("\n Creating pickle")
with open('trainingset_with_wmd.pickle', 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\n Creating csv")
df.to_csv('trainingset_with_wmd.csv')

print("\n Printing df.head as sample")
print(df.head())

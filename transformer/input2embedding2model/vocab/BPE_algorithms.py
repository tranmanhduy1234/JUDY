import pandas as pd

text = pd.read_csv(r"D:\chuyen_nganh\NT-HTKT\transformer\input2embedding2model\entext.txt",header=None)

corpus=[]
for row in text.values:
    tokens = row[0].split(" ")
    for token in tokens:
        corpus.append(token)

vocab = list(set(" ".join(corpus)))
vocab.remove(' ')

corpus = [" ".join(token) for token in corpus]
corpus=[token+' </w>' for token in corpus]

import collections
corpus = collections.Counter(corpus)
corpus = dict(corpus)

def get_stats(corpus):
    pairs = collections.defaultdict(int)
    for word, freq in corpus.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

import re
def merge_vocab(pair, corpus_in):
    corpus_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for word in corpus_in:
        w_out = p.sub(''.join(pair), word)
        corpus_out[w_out] = corpus_in[word]
    
    return corpus_out

pairs = get_stats(corpus)
best = max(pairs, key=pairs.get)
print("Max frequent pair: ", best)
print("="*100)
corpus = merge_vocab(best, corpus)
print("After Merging: ", corpus)

best = "".join(list(best))

merges = []
merges.append(best)
vocab.append(best)

num_merges = 10
for i in range(num_merges):
    
    pairs = get_stats(corpus)
    
    best = max(pairs, key=pairs.get)
    
    corpus = merge_vocab(best, corpus)
    
    merges.append(best)
    vocab.append(best)

merges_in_string = ["".join(list(i)) for i in merges]
print("BPE Merge Operations:",merges_in_string)
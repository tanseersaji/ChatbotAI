import pandas as pd
import nltk
import itertools
import pickle

lines = pd.read_csv('lines.csv')
lines = lines.drop(['Unnamed: 0'],axis=1)
lines = lines.dropna()

lines = list(lines['subt'])
lines = ['BOS '+l+' EOS' for l in lines]

text = ' '.join(lines)

vocab = list(set(list(text)))

print(len(vocab),"unique chars found.")

with open('vocabs.txt', 'w') as v:
    for c in vocab:
        v.write(c+'\n')
import pandas as pd
import time

t0 = time.time()

lines = pd.read_csv('lines.csv')
lines = lines.drop(['Unnamed: 0'],axis=1)
lines = lines.dropna()

corpus = pd.DataFrame(columns=['context','reply'])
context,reply = [],[]

lastContext = None

for _,line in lines.iterrows():
    sub = line['subt'].lower()
    if lastContext is None:
        lastContext = sub
    else:
        context.append(lastContext)
        reply.append(sub)
        lastContext = sub

corpus['context'] = context
corpus['reply'] = reply

print(corpus.head())

corpus.to_csv('corpus.csv')

print("Corpus created in {}s".format(time.time() - t0))
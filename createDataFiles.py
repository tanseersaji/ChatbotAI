import pandas as pd
corpus = pd.read_csv('corpus.csv')

reply = open('reply','a',encoding='utf8')
context = open('context','a',encoding='utf8')

for _,row in corpus.iterrows():
    context.write(row['context']+'\n')
    reply.write(row['reply']+'\n')
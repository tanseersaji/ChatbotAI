import os
import re
import pysubs2 as pysubs
import pandas as pd

subtitles = os.listdir('./datasets')
print(len(subtitles), "Files loaded")

lines = []

def format(subt):
    subt = subt.replace('\\N',' ')
    subt = re.sub(r"\{(.*?)\}",'',subt)
    subt = subt.replace('.','')
    subt = re.sub(r"[^a-zA-Z\s:]",'',subt)
    subt = re.sub(r"[^\S ]+","",subt)
    return subt

for file in subtitles:
    try:
        subs = pysubs.load('./datasets/'+file,encoding=u'utf-8')
    except:
        try:
            subs = pysubs.load('./datasets/'+file,encoding='cp1252')
        except:
            continue 
    print(file)
    for line in subs:
        subt = format(line.text)
        lines.append({"subt":subt})

dataframe = pd.DataFrame(data=lines)
dataframe = dataframe.drop_duplicates()
dataframe.to_csv('lines.csv')
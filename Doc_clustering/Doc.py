from __future__ import print_function
import nltk
import re

titles=[]
inputs=[]
w=[]
fr=[]

stopwords=nltk.corpus.stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer("english")

def tokenize_and_stem(text):
    tokens=[word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens=[]
    for token in tokens:
        if re.search('[a-zA-Z]',token):
            filtered_tokens.append(token)
    stems=[stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    tokens=[word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens=[]
    for token in tokens:
        if re.search('[a-zA-z]',token):
            filtered_tokens.append(token)
    return filtered_tokens

def comparing():
    totalvocab_stemmed=[]
    totalvocab_tokenized=[]
    for i in inputs:
        allwords_stemmed=tokenize_and_stem(i)
        totalvocab_stemmed.extend(allwords_stemmed)
        allwords_tokenized=tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)   

    description=[]
    for i in inputs:
        m=tokenize_and_stem(i)
        description.append(m)
    group=[]
    for i in inputs:
        text=([word for (word,pos) in nltk.pos_tag(nltk.word_tokenize(i)) if pos[0] == 'N'])
        group.append(text)
    names=[]
    for i in range(len(inputs)):
        res = max(set(group[i]) , key = group[i].count)
        names.append(res)
    names 

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df = 0.8,max_features=200000, min_df=0.2,stop_words='english',use_idf=True,tokenizer=tokenize_and_stem,
                       ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(inputs)

    from sklearn.metrics.pairwise import cosine_similarity
    dist = 1-cosine_similarity(tfidf_matrix)

    import numpy as np
    import pandas as pd
    from scipy.cluster import hierarchy
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    X=np.array(dist)
    frame=pd.DataFrame(dist)

    X=frame
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled=scaler.fit_transform(X)

    from sklearn.metrics import silhouette_score
    km_scores= []
    km_silhouette = []
    for i in range(2,len(dist)):
        km = KMeans(n_clusters=i, random_state=0).fit(X_scaled)
        preds = km.predict(X_scaled)
    
        km_scores.append(-km.score(X_scaled))
    
        silhouette = silhouette_score(X_scaled,preds)
        km_silhouette.append(silhouette)
        print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))

    d=km_silhouette.index(max(km_silhouette))
    k_value=d+2
    
    num_clusters = k_value
    km=KMeans(n_clusters = num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()

    groups={'Title':titles,'cluster':clusters,}
    data_frame=pd.DataFrame(groups,index=[clusters],columns= ['Title','cluster'])
    p=data_frame.values.tolist()
    w.append(p)
    MyTkTextBox.insert(END,"Titles\tClusters\n")
    for i in w:
        MyTkTextBox.insert(END,i)
    print(data_frame)
    MyTkTextBox.insert(END,"\n\nThe different groups are:")
    result=[]
    for i in range(num_clusters):
        text1="\nGroup ",i," names:\n"
        str = convertTuple(text1)
        result.append(str)
        for title in data_frame.loc[i]['Title'].values.tolist():
            text1="  %s,"% title
            str = convertTuple(text1)
            result.append(str)

    for i in result:
        MyTkTextBox.insert(END,i)

    MyTkTextBox.insert(END,"\n\n The Text Documents which something related to \n")
    rel=[]
    for i in range(len(names)):
        text="document ",i,"--"+names[i],"\n"
        str=convertTuple(text)
        rel.append(str)
    for i in rel:
        MyTkTextBox.insert(END,i)

def convertTuple(tup):
    st = ''.join(map(str, tup))
    return st

import tkinter as tk
from tkinter import*
from tkinter import ttk

fc = Tk()
fc.config(bg="purple")
frame=Frame(fc,height=700,width=1500,bg="light yellow",highlightcolor="black",highlightthickness="20")
L1 = Label(fc, text="NAME",width=10,font="times")
L1.place(x=170,y=100)

def tinp():
    inp = tl.get(1.0, "end-1c")
    titles.append(inp)
    print(titles)

def clear1():
    tl.delete(1.0,END)

def clear2():
    inputtxt.delete(1.0,END)

tl= Text(fc, bd =5,width=40,height=1,font="bold")
tl.place(x=270,y=100)
name =Label(fc,text="INPUTS",font="times").place(x=650,y=150)
#title_button=Button(fc,command=tinp,text="submit",font="times",width=10).place(x=680,y=95)
#clrbtn1 = tk.Button(fc,text = "Add", command = clear1,width=10,font="times").place(x=780,y=95)

def printInput1():
    inp1 = inputtxt.get(1.0, "end-1c")
    lbl.config(text = "Provided Input: "+inp1)
    inputs.append(inp1)
    print(inputs)

# TextBox Creation
inputtxt = tk.Text(fc,height = 10,width = 120)
inputtxt.place(x=200,y=180)
printButton = tk.Button(fc,text = "SUBMIT", command = lambda:[tinp(),printInput1()],font="times",width=10).place(x=600,y=350)
clrbtn2 = tk.Button(fc,text = "CLEAR", command = lambda:[clear1(),clear2()],font="times",width=10).place(x=700,y=350)
lbl = tk.Label(fc, text = "",width=138)
lbl.place(x=200,y=400)
button = tk.Button(fc, text="Click here to view Result", bg='White', fg='Black',command=comparing,font="times")
button.place(x=615,y=670)

V = ttk.Scrollbar(fc)
V.pack(side = RIGHT, fill = Y)
style = ttk.Style()
style.configure('TScrollbar', background='#E66565', troughcolor='#E0F5DA', arrowcolor='#65E6A9', highlightcolor='#E0F5DA') 
V.configure(style = 'TScrollbar')
MyTkTextBox = Text(fc, yscrollcommand = V.set,font="bold",highlightcolor="black")
MyTkTextBox.config(wrap = NONE, width = 107, height = 10)
MyTkTextBox.place(x=200,y=450)
V.config(command = MyTkTextBox.yview)
frame.pack(pady=40)
fc.mainloop()

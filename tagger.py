from tkinter import *
import nltk
from nltk.tag import CRFTagger
import pycrfsuite
from trainer import sent2features

root = Tk()

inputLabel = Label(root, text='Masukkan kalimat:')
inputLabel.pack()
sentInput = Entry(root, width =200)
sentInput.pack()
tagButt = Button(root, text='Tag it', command=lambda:getPosTag())
tagButt.pack()
perLabel = Label(root, text='')
perLabel.pack()
jobLabel = Label(root, text='')
jobLabel.pack()
subLabel = Label(root, text='')
subLabel.pack()
orgLabel = Label(root, text='')
orgLabel.pack()
geoLabel = Label(root, text='')
geoLabel.pack()

def getPosTag():
    global perLabel,jobLabel,subLabel,orgLabel,geoLabel
    raw_sent = sentInput.get()
    ct = CRFTagger()
    ct.set_model_file('all_indo_man_tag_corpus_model.crf.tagger')
    
    tokens = nltk.tokenize.word_tokenize(raw_sent)
    postagged = ct.tag_sents([tokens])
    
    data = []
    for token in postagged[0]:
        data.append(token+('O',))
    
    tagger_ner = pycrfsuite.Tagger()
    tagger_ner.open('model_ner.crfsuite')
    ner = tagger_ner.tag(sent2features(data,False))
    
    for i in range(len(ner)):
        data[i] = data[i][0:2]+(ner[i],)
    
    tagger_oh = pycrfsuite.Tagger()
    tagger_oh.open('model_oh.crfsuite')
    oh = tagger_oh.tag(sent2features(data,True))
    
    for i in range(len(oh)):
        data[i] += (oh[i],)
    
    per = []
    job = []
    sub = []
    org = []
    geo = []
    
    for token in data:
        if token[3] == '1':
            label = token[2][-3:]
            if label == 'PER':
                per.append(token[0])
            elif label == 'ORG':
                org.append(token[0])
            elif label == 'SUB':
                sub.append(token[0])
            elif label == 'JOB':
                job.append(token[0])
            elif label == 'GEO':
                geo.append(token[0])
    perLabel.config(text='PER: '+(' ').join(per))
    jobLabel.config(text='JOB: '+(' ').join(job))
    subLabel.config(text='SUB: '+(' ').join(sub))
    orgLabel.config(text='ORG: '+(' ').join(org))
    geoLabel.config(text='GEO: '+(' ').join(geo))

root.mainloop()
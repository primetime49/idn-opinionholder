import csv
from nltk.tag import CRFTagger
import pycrfsuite
import time
import nltk
import random

def word2features(sent, i, oh):
    word = sent[i][0]
    postag = sent[i][1]
    label = sent[i][2]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word[:3]=' + word[:3],
        'word[:2]=' + word[:2],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
    ]
    if oh == True:
        features.extend(['label=' + label,])
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        label1 = sent[i-1][2]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
        ])
        if oh == True:
            features.extend(['-1:label=' + label1,])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        label1 = sent[i+1][2]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
        ])
        if oh == True:
            features.extend(['+1:label=' + label1,])
    else:
        features.append('EOS')
                
    return features

def sent2features(sent, oh):
    return [word2features(sent, i, oh) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label, oh in sent]

def sent2oh(sent):
    return [oh for token, postag, label, oh in sent]

def sent2tokens(sent):
    return [token for token, postag, label, oh in sent]

def getData(filename):
    ct = CRFTagger()
    ct.set_model_file('all_indo_man_tag_corpus_model.crf.tagger')
    
    result = []
    annotated = []
    with open(filename+'.csv', 'r') as f:
        reader = csv.reader(f)
        annotated = list(reader)
        
    sent = []
    sent_gold = []
    sent_oh = []
    curr_sent = ''
    for token in annotated:
        if curr_sent != str(token[1])+' '+str(token[2]):
            hasil = ct.tag_sents([sent])
            mytuple = []
            for idx in range(len(sent)):
                try:
                    mytuple.append(hasil[0][idx]+(sent_gold[idx],sent_oh[idx]))
                except IndexError:
                    pass
            result.append(mytuple)
            sent = []
            sent_gold = []
            sent_oh = []
            curr_sent = str(token[1])+' '+str(token[2])
        sent.append(token[4])
        sent_gold.append(token[5])
        sent_oh.append(token[6])
    hasil = ct.tag_sents([sent])
    mytuple = []
    for idx in range(len(sent)):
        try:
            mytuple.append(hasil[0][idx]+(sent_gold[idx],sent_oh[idx]))
        except:
            pass
    result.append(mytuple)
    result = result[1:]
    print('Total sentence: '+str(len(result)))
    random.shuffle(result)
    return result

def getTrainData(data,start,finish):
    X_train_ner = [sent2features(s, False) for s in data[start:finish]]
    y_train_ner = [sent2labels(s) for s in data[start:finish]]

    X_train_oh = [sent2features(s, True) for s in data[start:finish]]
    y_train_oh = [sent2oh(s) for s in data[start:finish]]
    
    return (X_train_ner,y_train_ner,X_train_oh,y_train_oh)

def trainNER(X_train_ner, y_train_ner):
    trainer_ner = pycrfsuite.Trainer(verbose=False)

    t0= time.clock()

    for xseq, yseq in zip(X_train_ner, y_train_ner):
        trainer_ner.append(xseq, yseq)
        
    trainer_ner.set_params({
        'c1': 1e-2,   # coefficient for L1 penalty
        'c2': 1e-2,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
        'feature.possible_transitions': True # include transitions that are possible, but not observed
    })

    trainer_ner.train('model_ner.crfsuite')

    t1 = time.clock()

    print("Time elapsed for training NER: %.2f seconds" % (t1 - t0))
    
    return trainer_ner

def trainOH(X_train_oh,y_train_oh):
    trainer_oh = pycrfsuite.Trainer(verbose=False)

    t0= time.clock()

    for xseq, yseq in zip(X_train_oh, y_train_oh):
        trainer_oh.append(xseq, yseq)
        
    trainer_oh.set_params({
        'c1': 1e-2,   # coefficient for L1 penalty
        'c2': 1e-2,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
        'feature.possible_transitions': True # include transitions that are possible, but not observed
    })

    trainer_oh.train('model_oh.crfsuite')

    t1 = time.clock()

    print("Time elapsed for training OH: %.2f seconds" % (t1 - t0))
    
    return trainer_oh

def tagNER(begin,end,data):
    tagger_ner = pycrfsuite.Tagger()
    tagger_ner.open('model_ner.crfsuite')
    tested_ner = []
    for i in range (begin,end):
        tested_ner.append(tagger_ner.tag(sent2features(data[i],False)))
    return tested_ner

def tagOH(begin,end,data):
    tagger_oh = pycrfsuite.Tagger()
    tagger_oh.open('model_oh.crfsuite')
    tested_oh = []
    for i in range (begin,end):
        tested_oh.append(tagger_oh.tag(sent2features(data[i],True)))
    return tested_oh

def evaluateData(data,begin,end,tested_ner,tested_oh):
    goldTest = data[begin:end]
    recall = 0
    num_words = 0

    for i in range(len(goldTest)):
        sentence = goldTest[i]
        for j in range(len(sentence)):
            token = sentence[j]
            oh = token[3]
            if oh == '1':
                num_words += 1
                if tested_ner[i][j] != 'O':
                    recall += 1
                #else:
                    #print(goldTest[i][j][0]+' failed to be tagged')
                    
    print("Recall: %.2f%%" % (recall/num_words*100))
    recall = recall/num_words*100

    precision = 0
    num_words = 0

    for i in range(len(tested_oh)):
        sentence = tested_oh[i]
        for j in range(len(sentence)):
            oh = sentence[j]
            if oh != '0' and tested_ner[i][j] != 'O':
                num_words += 1
                if tested_ner[i][j][-3:] == goldTest[i][j][2][-3:]:
                    precision += 1
                #else:
                    #print(goldTest[i][j][0]+' predicted as '+tested_ner[i][j][-3:]+', should be '+goldTest[i][j][2][-3:])

    print("Precision: %.2f%%" % (precision/num_words*100))
    precision = precision/num_words*100

    f1 = 2*recall*precision/(recall+precision)
    print("F1 measure: %.2f%%" % f1)
    return (recall,precision,f1)

def trainData(data,X_train_ner,y_train_ner,X_train_oh,y_train_oh,begin,end):
    trainer_ner = trainNER(X_train_ner,y_train_ner)
    tested_ner = tagNER(begin,end,data)
    trainer_oh = trainOH(X_train_oh,y_train_oh)
    tested_oh = tagOH(begin,end,data)
    return evaluateData(data,begin,end,tested_ner,tested_oh)
    #getOH(data,begin,end,tested_ner,tested_oh)

def getOH(data,begin,end,tested_ner,tested_oh):
    goldTest = data[begin:end]
    for i in range(len(tested_oh)):
        sentence = tested_oh[i]
        per = []
        job = []
        sub = []
        org = []
        geo = []
        oh_here = False
        for j in range(len(sentence)):
            oh = sentence[j]
            if oh != '0':
                oh_here = True
                if tested_ner[i][j][-3:] == 'PER':
                    per.append(goldTest[i][j][0])
                elif tested_ner[i][j][-3:] == 'JOB':
                    job.append(goldTest[i][j][0])
                elif tested_ner[i][j][-3:] == 'SUB':
                    sub.append(goldTest[i][j][0])
                elif tested_ner[i][j][-3:] == 'ORG':
                    org.append(goldTest[i][j][0])
                elif tested_ner[i][j][-3:] == 'GEO':
                    geo.append(goldTest[i][j][0])
        tokens = []
        if oh_here == True:
            for j in range(len(sentence)):
                tokens.append(goldTest[i][j][0])
            raw_sent = " ".join(tokens)
            print(raw_sent)
            print('PER: '+" ".join(per))
            print('JOB: '+" ".join(job))
            print('SUB: '+" ".join(sub))
            print('ORG: '+" ".join(org))
            print('GEO: '+" ".join(geo))

def main():
    filename = input('Dataset filename (without .csv): ')
    split_train = int(input('Percentage of training data (without %) [e.g. 70,75,80]: '))
    # Code below is used for evaluation
    '''iter = int(input('How many iteration? '))
    f1_total = 0
    rec_total = 0
    prec_total = 0
    time_total = 0
    for i in range(iter):
        t0= time.clock()
        result = getData(filename)
        totalData = len(result)
        start = 0
        finish = int(totalData*split_train/100)
        X_train_ner,y_train_ner,X_train_oh,y_train_oh = getTrainData(result,start,finish)
        rec,prec,f1_score = trainData(result,X_train_ner,y_train_ner,X_train_oh,y_train_oh,finish,totalData)
        f1_total += f1_score
        rec_total += rec
        prec_total += prec
        t1 = time.clock()
        time_total += (t1-t0)
    f1 = f1_total/iter
    rec = rec_total/iter
    prec = prec_total/iter
    time_avg = time_total/iter
    print('-------------')
    print("average recall score: %.2f%%" % rec)
    print("average precision score: %.2f%%" % prec)
    print("average F1 score: %.2f%%" % f1)
    print("average time elapsed for training: %.2f seconds" % (time_avg))
    print('-------------')'''
    # Code above is used for evaluation
    result = getData(filename)
    totalData = len(result)
    start = 0
    finish = totalData
    X_train_ner,y_train_ner,X_train_oh,y_train_oh = getTrainData(result,start,finish)
    trainer_ner = trainNER(X_train_ner,y_train_ner)
    trainer_oh = trainOH(X_train_oh,y_train_oh)
    print('Models are saved to: model_ner.crfsuite and model_oh.crfsuite')
    
if __name__== "__main__":
    main()
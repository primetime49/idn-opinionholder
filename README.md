# idn-opinionholder

Identification of Opinion Holder in Indonesian Sentence

## Getting started
### Prequisites

Python 3.6.4
* nltk==3.3
* python-crfsuite==0.9.5
* tkinter==8.6

Windows 10

## How to run
### Annotator program

Run 'python oh_annotator.py' on cmd. Write the dataset filename when prompted. Make sure the path to the file is right. You’re good to go. Annotated sentences are appended onto a file with the same name, but ending with '-annotated', if not already.

### Trainer

Run 'python trainer.py' on cmd. Write the annotated dataset filename (without .csv) and the percentage of training data when prompted. You’re good to go. The models will be saved in the same directory.

### Tagger

Run 'python tagger.py' on cmd. In the GUI afterwards, you’ll see a text field where you’ll put the sentence to be tagged. Once done, click 'Tag it', the OH label will be shown below. To change the sentence, just re-type it on the text field and hit the button again. Voila!

## Dataset
### Extending the dataset

Because the model cannot be retrained, all you can do is extend the dataset instead of retraining the model. To extend the dataset, just append the new annotated tokens onto your existing dataset, and then train it again using the trainer. Make sure your file stays .csv otherwise bye bye.

### Dataset Format

* Make sure your original dataset is formatted to csv with columns: datetime, doc_id, sent_idx, sent
* The annotated dataset will have columns: datetime, doc_id, sent_idx, token_idx, token, ner_label, oh_label

## Acknowledgments

Big thanks to:
* [POS Tagger](https://github.com/famrashel/idn-tagged-corpus) - POS Tagger for Indonesian Language
* [python-crfsuite](https://github.com/scrapinghub/python-crfsuite) - CRF implementation in Python

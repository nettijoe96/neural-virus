import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
import sys


############################################
### Load All Papers - Cleaned and Spaced
############################################

df = pd.read_csv('spacy.csv',index_col=0)


############################################
### Load Paper Categories Hashes
############################################

paper_fns = ['positive_out.txt', 'negative_out.txt']
if len(sys.argv) > 1:
	paper_fns = sys.argv[1:]
	
positive_hash = None
with open(paper_fns[0], 'r') as f:
	positive_hash = f.read().split('\n')
	
negative_hash = None
with open(paper_fns[1], 'r') as f:
	negative_hash = f.read().split('\n')
	

############################################
### Load Papers from Hashes
############################################

hashes = df['paper_id'].values.tolist()
paper_text = df['processed_text'].values.tolist()

positive_papers = []

for hash in positive_hash:
	for idx,hash2 in enumerate(hashes):
		if hash == hash2:
			try:
				positive_papers.append(paper_text[idx])
				break
			except:
				print(hash)
				
negative_papers = []
for hash in negative_hash:
	for idx,hash2 in enumerate(hashes):
		if hash == hash2:
			try:
				negative_papers.append(paper_text[idx])
				break
			except:
				print(hash)
			
print(len(positive_papers))
print(len(negative_papers))
#quit()


############################################
### Create Vocabulary from Papers
############################################

from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

# turn a doc into clean tokens
def clean_doc_vocab(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc and add to vocab
def add_doc_to_vocab(doc, vocab):
	tokens = clean_doc_vocab(doc)
	# update counts
	vocab.update(tokens)

# load all docs in a directory
def process_docs_vocab(doc_list, vocab):
		for doc in doc_list:
			add_doc_to_vocab(doc, vocab)

# define vocab
vocab = Counter()
# add all docs to vocab
process_docs_vocab(positive_papers, vocab)
process_docs_vocab(negative_papers, vocab)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))


# keep tokens with a min occurrence
min_occurane = 1000
vocab = [k for k,c in vocab.items() if c >= min_occurane]
print(len(vocab))
vocab = set(vocab)

#vocab = set([k for k,c in vocab.most_common(10)])

#quit()

############################################
### Load Train, Validation, and Test sets
############################################

from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# turn a doc into clean tokens
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens


# load all docs in a directory
def process_docs(doc_list, vocab):
	documents = list()
	for doc in doc_list:
		tokens = clean_doc(doc, vocab)
		# add to list
		documents.append(tokens)
	return documents

# 70% Train
# 20% Validation
# 10% Test
train_split = 0.7
valid_split = 0.9

############################################
### Training Set
############################################

# load all training reviews
positive_docs = process_docs(positive_papers[:int(len(positive_papers)*train_split)], vocab)
negative_docs = process_docs(negative_papers[:int(len(negative_papers)*train_split)], vocab)
train_docs = negative_docs + positive_docs

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = array([0 for _ in range(len(positive_docs))] + [1 for _ in range(len(negative_docs))])


############################################
### Validation Set
############################################

# load all test reviews
positive_docs = process_docs(positive_papers[int(len(positive_papers)*train_split):int(len(positive_papers)*valid_split)], vocab)
negative_docs = process_docs(negative_papers[int(len(negative_papers)*train_split):int(len(negative_papers)*valid_split)], vocab)
valid_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(valid_docs)
# pad sequences
Xvalid = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
yvalid = array([0 for _ in range(len(positive_docs))] + [1 for _ in range(len(negative_docs))])


############################################
### Test Set
############################################

# load all test reviews
positive_docs = process_docs(positive_papers[int(len(positive_papers)*valid_split):], vocab)
negative_docs = process_docs(negative_papers[int(len(negative_papers)*valid_split):], vocab)
valid_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(valid_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(len(positive_docs))] + [1 for _ in range(len(negative_docs))])


############################################
### Run 1D CNN
############################################
print(Xtrain[-1])
print(len(Xtrain))
print(len(Xvalid))
print(Xvalid[-1])
print(len(Xtest))
print(Xtest[-1])

print(ytrain)
print(len(ytrain))
print(yvalid)
print(len(yvalid))
print(ytest)
print(len(ytest))

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=100, validation_data=(Xvalid, yvalid), verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))




















































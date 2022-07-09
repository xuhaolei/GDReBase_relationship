import nltk
from nltk.tokenize import RegexpTokenizer
import pickle
with open('vocab.pkl','rb') as f:
	vocab = pickle.load(f)
print(len(vocab))
print(vocab)
''' 
    Code to train word2vec model on phrase tagged corpus
    Ensure that the relevant details are entered in config.ini.    
    Then, execute by running `python3 train_phrase2vec.py`
'''

from nltk import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import configparser

config = configparser.ConfigParser()
try:
    config.read('config.ini')
except:
    print ("ERROR: No config file. Create a new file called config.ini")
    exit()

try:
    embeddings_file = config['DEFAULT']['embeddings_file']
except:
    print ("ERROR: No output embeddings file specified. Check config.ini")
    exit()

try:
    corpus = config['DEFAULT']['corpus']
except:
    print ("ERROR: No corpus specified. Check config.ini")
    exit()

text = open(corpus, "r").read()
sentences = sent_tokenize(text)
sents = [word_tokenize(sentence) for sentence in sentences]

model = Word2Vec(sents, min_count=1, size=300)

f = open(embeddings_file, "w+")

cnt = 0
for term in model.wv.vocab:
    if not term.strip():
        continue
    if cnt == 0:
        string = " ".join([term.strip(), " ".join([str(s) for s in model.wv[term]])])
        f.write(string)
    else:
        f.write("\n" + " ".join([term.strip(), " ".join([str(s) for s in model.wv[term]])]))
    cnt = 1

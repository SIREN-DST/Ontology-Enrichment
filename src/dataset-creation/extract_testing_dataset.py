''' 
    Step 3.3 of dataset extraction pipeline: Code to extract possible term pairs from a test corpus
    Ensure that the relevant details are entered in config.ini.    
    Then, execute by running `python3 extract_testing_dataset.py` 
    The concepts are saved in a file called `test_alt.tsv` in the `train_tf` and `train_multi` directories created in stage 3.1.
'''


import re, configparser
import urllib.request
import spacy, neuralcoref, itertools
from bs4 import BeautifulSoup
from bs4.element import Comment
from subject_verb_object_extract import findSVOs, nlp
from nltk.chunk.regexp import RegexpParser
from nltk import pos_tag, word_tokenize
from nltk.tree import Tree

config = configparser.ConfigParser()
try:
    config.read('config.ini')
except:
    print ("ERROR: No config file. Create a new file called config.ini")
    exit()

try:
    url = config['extract_testing_dataset']['url']
except:
    print ("ERROR: No URL field specified. Check config.ini")
    exit()

try:
    domainName = config['DEFAULT']['domain']
except:
    print ("ERROR: No domain specified. Check config.ini")
    exit()


def getInstances(text):
    grammar = """
        PRE:   {<NNS|NNP|NN|NP|JJ|UH>+}
        INSTANCE:   {(<JJ+>)?<PRE>}
    """
    chunker = RegexpParser(grammar)
    taggedText = pos_tag(word_tokenize(text))
    textChunks = chunker.parse(taggedText)
    current_chunk = []
    for i in textChunks:
        if (type(i) == Tree and i.label() == "INSTANCE"):
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
    return current_chunk


html = urllib.request.urlopen(url)
soup = BeautifulSoup(html, "lxml")
data = soup.findAll("p")
paras = [o.text for o in data]

nlp = spacy.load('en_core_web_lg')

# load NeuralCoref and add it to the pipe of SpaCy's model, for coreference resolution
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')

paras = [nlp(para)._.coref_resolved for para in paras]

testData = []
for para in paras:
    instances = getInstances(para)
    ls = list(set([a.lower() for a in instances]))
    ls = list(set(list(itertools.combinations(ls, 2))))
    print (ls)
    testData.extend(["\t".join([a,b]) for (a,b) in ls])
    
testData = list(set(testData))

trueData = [el + ["false"] for el in testData]
multiData = [el + ["none"] for el in testData] 

open("train_tf/test_alt.tsv", "a+").write("\n" + trueFalse)
open("train_multi/test_alt.tsv", "a+").write("\n" + multiFalse)


open(domainName + "_testing.tsv","w+").write("\n".join(testData))

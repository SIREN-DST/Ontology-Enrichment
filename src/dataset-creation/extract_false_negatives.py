''' 
    Step 3.2 of dataset extraction pipeline: Code to generate false negatives for a sentence
    Ensure that the relevant details for dataset, model and domain are entered in config.ini.
    Then, execute by running `python3 extract_false_negatives.py` 
    The false negatives are directly added to the dataset created in stage 3.1.
'''


from bs4 import BeautifulSoup
import requests, configparser
from nltk.tree import Tree
from nltk.chunk.regexp import RegexpParser
from nltk import pos_tag, word_tokenize
from gensim.models import KeyedVectors
from SPARQLWrapper import SPARQLWrapper, JSON

 # Name of the word2vec model used to compute similarity
# Name of the output file containing false negatives
config = configparser.ConfigParser()
try:
    config.read('config.ini')
except:
    print ("ERROR: No config file. Create a new file called config.ini")
    exit()

try:
    original_dataset = config['extract_false_negatives']['dataset']
except:
    print ("ERROR: No dataset specified. Check config.ini")

try:
    model = config['extract_false_negatives']['model']
except:
    print ("ERROR: No word2vec model specified. Check config.ini")

try:
    domainName = config['DEFAULT']['domain']
except:
    print ("ERROR: No domain specified. Check config.ini")


def generateFalseRelations(termlist):
    final_list = []
    idx = 0
    for (a, b, label) in termlist:
        if label == "none" or label == "hypernym":
            termname = a
            hypernym = b
        elif label == "hyponym" or label == "synonym":
            termname = b
            hypernym = a
            
            
        relatedWords = [hypernym]

        q = "_".join(termname.lower().split(" "))
        q = q[0].upper() + q[1:]
        url = "https://en.wikipedia.org/wiki/" + q
        
        r = requests.get(url)
        soup = BeautifulSoup(r.content) 
        classes = soup.findAll("div", {"class":"mw-parser-output"})
        text = ""
        if classes:
            paras = classes[0].findAll("p")
            if paras:
                for word in relatedWords:
                    for para in paras:
                        paratext = para.text.lower()
                        if word.lower() in paratext and termname.lower() in paratext:
                            text += paratext + "\n"
            else:
                continue
        else:
            continue
        tagged = pos_tag(word_tokenize(text))
        relatedWords = [l.lower() for l in relatedWords]
        allConcepts = [elem[0] for elem in tagged if elem[1] == "NN" and isValid(elem[0]) and elem[0].lower() not in relatedWords]
        newList = []
        for concept in allConcepts:
            try:
                sim = 1/model.similarity("_".join(hypernym.split(" ")).lower(), "_".join(concept.split(" ")).lower())
            except:
                sim = 1

            newList.append((termname.lower(), concept.lower(), str(sim)))
        print (newList)
        string = "\n".join(["\t".join(elem) for elem in newList])
        f.write(string)
        f.write("\n")
        final_list.extend(newList)
        
    return final_list

def isValid(s):
    if not s.isalnum():
        return False
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True



def processString(ls):
    return [" ".join(string.lower().split("_")) for string in ls]

allRelations = [l.split("\t") for l in open(original_dataset, "r").read().split("\n")]
concepts = [processString(l[:3]) for l in allRelations]
falseRelations = generateFalseRelations_final(concepts)

falseRelations.sort(key=lambda x: float(x[2]))

multiFalse = ["\t".join(l[:2] + ["none"]) for l in falseRelations]
trueFalse = ["\t".join(l[:2] + ["false"]) for l in falseRelations]

open("train_tf/train.tsv", "a+").write("\n" + trueFalse)
open("train_multi/train.tsv", "a+").write("\n" + multiFalse)

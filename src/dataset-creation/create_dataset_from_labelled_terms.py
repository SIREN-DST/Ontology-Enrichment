''' 
    Step 3.1 of dataset extraction pipeline: Code to create dataset from manually annotated DBPedia terms
    Ensure that the relevant details are entered in config.ini.
    Then execute by running `python3 create_dataset_from_labelled_terms.py`
    The T/F datasets are created in a directory called train_tf and multi-class datasets are created in a directory called train_multi.
'''


import random, subprocess, configparser

config = configparser.ConfigParser()
try:
    config.read('config.ini')
except:
    print ("ERROR: No config file. Create a new file called config.ini")
    exit()

try:
    annotated_terms_name = config['extract_concepts_from_ontology']['annotated_terms_name']
except:
    print ("ERROR: No ontology field specified. Check config.ini")
    exit()

try:
    domainName = config['DEFAULT']['domain']
except:
    print ("ERROR: No domain specified. Check config.ini")
    exit()


allLines = [line.split("\t") for line in open(annotated_terms_name, "r").read().split("\n")]
def processString(ls):
    return [" ".join(string.lower().split("_")) for string in ls]

tfLines = [processString(l[:2]) + ["true" if el=="T" else "false" for el in list(l[-1])] for l in allLines]
relationLines = [processString(l[:2]) + ["none" if l[-1]=="F" else el.lower() for el in [l[2]]] for l in allLines]

def trainTestSplit(allRelations, noneStr, ratio):
    noneRelations = [el for el in allRelations if el[-1] == noneStr]
    nonNullRelations = [el for el in allRelations if el[-1] != noneStr]

    valLen1 = int(ratio * len(noneRelations))
    valRelations1 = random.sample(noneRelations, valLen1)


    valLen2 = int(ratio * len(nonNullRelations))
    valRelations2 = random.sample(nonNullRelations, valLen2)

    valRelations = valRelations1 + valRelations2

    trainRelations1 = [relation for relation in noneRelations if relation not in valRelations1]
    trainRelations2 = [relation for relation in nonNullRelations if relation not in valRelations2]

    trainRelations = trainRelations1 + trainRelations2
    
    return (trainRelations, valRelations)
    
def trainTestValSplit(allRelations, noneStr):
    trainRelations, valRelations = trainTestSplit(allRelations, noneStr, 0.05)
    trainRelations, testRelations = trainTestSplit(trainRelations, noneStr, 0.1)
    return (trainRelations, testRelations, valRelations)

createDatasetDir = "mkdir train_tf train_multi"
process = subprocess.Popen(createDatasetDir.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

train, test, val = trainTestValSplit(tfLines, "false")
open("train_tf/train.tsv", "w+").write("\n".join(["\t".join(el) for el in train]))
open("train_tf/test.tsv", "w+").write("\n".join(["\t".join(el) for el in test]))
open("train_tf/val.tsv", "w+").write("\n".join(["\t".join(el) for el in val]))

train, test, val = trainTestValSplit(relationLines, "none")
open("train_multi/train.tsv", "w+").write("\n".join(["\t".join(el) for el in train]))
open("train_multi/test.tsv", "w+").write("\n".join(["\t".join(el) for el in test]))
open("train_multi/val.tsv", "w+").write("\n".join(["\t".join(el) for el in val]))
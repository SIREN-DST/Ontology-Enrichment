''' 
    Step 3 of dataset extraction pipeline: Code to create training and testing datasets
    from file of manually annotated DBPedia terms
    Ensure that the relevant details are entered in config.ini.
    Then execute by running `python3 create_dataset_from_labelled_terms.py`
    The datasets are stored in a directory called `dataset` inside `../../files/`.
'''


import random, subprocess, configparser, os

random.seed(0)

config = configparser.ConfigParser()
try:
    config.read('config.ini')
except:
    print ("ERROR: No config file. Create a new file called config.ini")
    exit()


try:
    domainName = config['DEFAULT']['domain']
except:
    print ("ERROR: No domain specified. Check config.ini")
    exit()

try:
    annotated_terms_name = os.path.abspath(config['dataset-creation']['annotated_terms_name']) + "/"
except:
    print ("ERROR: No ontology field specified. Check config.ini")
    exit()

allLines = [line.split("\t") for line in open(annotated_terms_name, "r").read().split("\n")]
def processString(ls):
    return [" ".join(string.lower().split("_")) for string in ls]

relationLines = [processString(l[:2]) + l[-1].lower() for l in allLines]

def train_test_split(allRelations, noneStr):
    ratio = 0.1

    noneRelations = [el for el in allRelations if el[-1] == noneStr]
    nonNullRelations = [el for el in allRelations if el[-1] != noneStr]

    testLen1 = int(ratio * len(noneRelations))
    testRelations1 = random.sample(noneRelations, testLen1)


    testLen2 = int(ratio * len(nonNullRelations))
    testRelations2 = random.sample(nonNullRelations, testLen2)

    testRelations = testRelations1 + testRelations2

    trainRelations1 = [relation for relation in noneRelations if relation not in testRelations1]
    trainRelations2 = [relation for relation in nonNullRelations if relation not in testRelations2]

    trainRelations = trainRelations1 + trainRelations2

    return (trainRelations, testRelations)

dataset_folder = "mkdir " + os.path.abspath("../../files") + "/dataset/"
process = subprocess.Popen(dataset_folder.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

train, test = train_test_split(relationLines, "none")
open(dataset_folder + "train.tsv", "w+").write("\n".join(["\t".join(el) for el in train]))
open(dataset_folder + "test.tsv", "w+").write("\n".join(["\t".join(el) for el in test]))
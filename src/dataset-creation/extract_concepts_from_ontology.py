''' 
	Step 1 of dataset extraction pipeline: Code to extract concepts from an ontology
	Ensure that the relevant details are entered in config.ini.
    Then, execute by running `python3 extract_concepts_from_ontology.py`.
	The concepts are saved in a file called `concepts_<domain_name>.txt` inside `../../files/`
'''


from pronto import Ontology
from re import finditer
import configparser, os


config = configparser.ConfigParser()
try:
    # Read config file
    config.read(os.path.abspath('../../config.ini'))
except:
    print ("ERROR: No config file. Create a new file called config.ini in root folder")
    exit()

try:
    # Ontologies to parse
    ontologies = config['dataset-creation']['ontologies'].split(", ")
except:
    print ("ERROR: No ontology field specified. Check config.ini")
    exit()

try:
    # Domain of ontology. Used for naming purposes
    domain = config['DEFAULT']['domain']
except:
    print ("ERROR: No domain specified. Check config.ini")
    exit()

# Folder for storing all input & output files
files_path = os.path.abspath("../../files/") + "/"

def split_by_camel_case(identifier):
    # Split string by camel-case
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return " ".join([m.group(0) for m in matches])

def extract_concepts_from_ontologies(ontologies):
    # Extracts concepts from ontologies, given a list of ontologies
    all_concepts = []
    for ontology in ontologies:
        concepts = [split_by_camel_case(term) for term in Ontology(ontology).terms.keys()]
        all_concepts.extend(concepts)
    return all_concepts

# Enter all ontologies you want parsed
concepts = list(set(extract_concepts_from_ontologies(ontologies)))

open(files_path + "concepts_" + domain + ".txt","w+").write("\n".join(concepts))

''' 
	Step 1 of dataset extraction pipeline: Code to extract concepts from an ontology
	Ensure that the relevant details are entered in config.ini.
    Then, execute by running `python3 extract_concepts_from_ontology.py`.
	The concepts are saved in a file called `concepts_<domain_name>.txt`
'''


from pronto import Ontology
from re import finditer
import configparser


config = configparser.ConfigParser()
try:
    config.read('config.ini')
except:
    print ("ERROR: No config file. Create a new file called config.ini")
    exit()

try:
    ontologies = config['extract_concepts_from_ontology']['ontology'].split(", ")
except:
    print ("ERROR: No ontology field specified. Check config.ini")
    exit()

try:
    domainName = config['DEFAULT']['domain']
except:
    print ("ERROR: No domain specified. Check config.ini")
    exit()


def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return " ".join([m.group(0) for m in matches])

def extractHypernymsFromOntology(ontology):    
    ont = Ontology(ontology)
    allConcepts = []
    listid = []
    dictelem = {}
    for term in ont:
        allConcepts.append(term)
        if term.children:
            a = str(term).split(":")
            b = a[0]
            listid.append(b[1:])
    for x in range(0,len(listid)):
        key = listid[x]
        try:
            if key in dictelem:
                child = ont[listid[x].children].split(":")
                ch = child[0]
                dictelem.get(key).append(ch[1:])
            else:
                childs = ont[listid[x]].children
                all_childs = ""
                for y in childs:
                    z = str(y).split(":")
                    f = z[0]
                    all_childs += f[1:]+","
                dictelem[key] = all_childs
        except:
            continue
    
    finalDict = {}

    for elem in dictelem:
        newelem = camel_case_split(elem)
        ls = dictelem[elem].split(",")[:-1]
        newval = ",".join([camel_case_split(el) for el in ls])
        finalDict[newelem] = newval

    hypernymsList = []
    for elem in finalDict:
        hypernymsList.extend([(elem, val) for val in finalDict[elem].split(",")])
    
    return (hypernymsList, allConcepts)

def parseOntologies(ontologies):
    allHypernyms, allConcepts = [], []    
    for ontology in ontologies:
        hypernyms, concepts = extractHypernymsFromOntology(ontology)
        allHypernyms.extend(hypernyms)
        allConcepts.extend(concepts)
    return (allHypernyms, allConcepts)

# Enter all ontologies you want parsed
hypernyms, words = parseOntologies(ontologies)

concepts = []
for word in words:
    if word.name:
        concepts.append(camel_case_split(word.name))
    else:
        concepts.append(camel_case_split(word.id.strip(":")))

concepts = list(set(concepts))

open("concepts_" + domainName + ".txt","w+").write("\n".join(list(set(concepts))))

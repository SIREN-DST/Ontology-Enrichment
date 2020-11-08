''' 
    Step 2 of dataset extraction pipeline: Code to extract related terms from DBPedia
    Ensure that the relevant details are entered in config.ini.
    Then, execute by running `python3 extract_terms_from_dbpedia.py`
    The terms from DBPedia are now stored in a file called
     `dbpedia_terms_<domain_name>.tsv` inside `../../files/`
'''

import configparser, os
from SPARQLWrapper import SPARQLWrapper, JSON

config = configparser.ConfigParser()
try:
    config.read('config.ini')
except:
    print ("ERROR: No config file. Create a new file called config.ini")
    exit()

# Folder for storing all input & output files
files_path = os.path.abspath("../../files/") + "/"

try:
    # Domain of ontology. Used for naming purposes
    domain = config['DEFAULT']['domain']
except:
    print ("ERROR: No domain specified. Check config.ini")
    exit()


concepts_file = files_path + "concepts_" + domain + ".txt"
concepts = open(concepts_file, "r").read().split("\n")
# Output file used for storing terms extracted from DBPedia
terms_file = files_path + "dbpedia_terms_" + domain + ".tsv"

def extract_related_terms_from_DBPedia(termlist):
    # Obtain related terms such as hypernyms and hyponyms from DBPedia
    final_list = []
    idx = 0
    for termname in termlist:
        tempList = []
        queryWord = "_".join(termname.split(" "))
        # Make SPARQL query for hypernyms
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setQuery("""SELECT * WHERE {<http://dbpedia.org/resource/"""+queryWord + """> <http://purl.org/linguistics/gold/hypernym> ?hypernyms .}""")
        idx+=1
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
        except Exception as e:
            print (idx, queryWord)
            print (e)
            continue

        if results["results"]["bindings"]:
            for result in results["results"]["bindings"]:
                res = result["hypernyms"]["value"]
                name = res.split('/')[-1]
                tempList.append([termname, name, "Hypernym"])
        else:
            termname2 = termname.lower()[0].upper() + termname.lower()[1:]
            queryWord2 = "_".join(termname2.split(" "))
            sparql = SPARQLWrapper("http://dbpedia.org/sparql")
            sparql.setQuery("""SELECT * WHERE {<http://dbpedia.org/resource/"""+queryWord2 + """> <http://purl.org/linguistics/gold/hypernym> ?hypernyms .}""")
            sparql.setReturnFormat(JSON)
            idx+=1
            try:
                results = sparql.query().convert()
            except Exception as e:
                print (idx, queryWord2)
                print (e)
                continue

            for result in results["results"]["bindings"]:
                res = result["hypernyms"]["value"]
                name = res.split('/')[-1]
                tempList.append([termname, name, "Hypernym"])
        
        # Make SPARQL query for hyponyms
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setQuery("""SELECT * WHERE {?hypernyms <http://purl.org/linguistics/gold/hypernym> <http://dbpedia.org/resource/"""+queryWord + """> .}""")
        sparql.setReturnFormat(JSON)
        idx+=1
        try:
#             print (queryWord)
            results = sparql.query().convert()
        except Exception as e:
            print (idx, queryWord)
            print (e)
            continue

        if results["results"]["bindings"]:
            for result in results["results"]["bindings"]:
                res = result["hypernyms"]["value"]
                name = res.split('/')[-1]
                tempList.append([name, termname, "Hyponym"])
        else:
            termname2 = termname.lower()[0].upper() + termname.lower()[1:]
            queryWord2 = "_".join(termname2.split(" "))
            sparql = SPARQLWrapper("http://dbpedia.org/sparql")
            sparql.setQuery("""SELECT * WHERE {?hypernyms <http://purl.org/linguistics/gold/hypernym> <http://dbpedia.org/resource/"""+queryWord2 + """> .}""")
            sparql.setReturnFormat(JSON)
            idx+=1
            try:
#                 print (queryWord2)
                results = sparql.query().convert()
            except Exception as e:
                print (idx, queryWord2)
                print (e)
                continue
            for result in results["results"]["bindings"]:
                res = result["hypernyms"]["value"]
                name = res.split('/')[-1]
                tempList.append([name, termname, "Hyponym"])
        
        appendingList = []
        for elem in tempList:
            if elem not in appendingList:
                appendingList.append(elem)
        final_list.extend(appendingList)
        
    return final_list

dbpedia_terms_unfiltered = extract_related_terms_from_DBPedia(concepts)

string = "\n".join(list(set(" ".join("\t".join(a).lower().split("_")) for a in 
                            [l for l in dbpedia_hypernyms_unfiltered if l[0]!=l[1]])))
open(terms_file, "w+").write(string)

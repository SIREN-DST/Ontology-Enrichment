''' 
    Step 2 of dataset extraction pipeline: Code to extract terms from DBPedia
    Ensure that the relevant details are entered in config.ini.
    Then, execute by running `python3 extract_terms_from_dbpedia.py`
    The relations from DBPedia are now stored in a file called `terms_from_dbpedia_<domain_name>.tsv`
'''

import configparser
from SPARQLWrapper import SPARQLWrapper, JSON

config = configparser.ConfigParser()
try:
    config.read('config.ini')
except:
    print ("ERROR: No config file. Create a new file called config.ini")
    exit()

try:
    conceptsName = config['extract_terms_from_dbpedia']['conceptsName']
except:
    print ("ERROR: No concepts file specified. Check config.ini")
    exit()

try:
    domainName = config['DEFAULT']['domain']
except:
    print ("ERROR: No domain specified. Check config.ini")
    exit()


concepts = open(conceptsName, "r").read().split("\n")

def dbpedia_parse(termlist):
    final_list = []
    idx = 0
    for termname in termlist:
        tempList = []
        queryWord = "_".join(termname.split(" "))
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
        
        queryWord = "_".join(termname.lower().split(" "))
        queryWord = queryWord[0].upper() + queryWord[1:]
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setQuery("""SELECT * WHERE {?synonyms <http://dbpedia.org/ontology/wikiPageRedirects> <http://dbpedia.org/resource/"""+ queryWord + """> .}""")
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
                res = result["synonyms"]["value"]
                name = res.split('/')[-1]
                tempList.append([termname, name, "Synonym"])

        else:
            termname2 = termname.lower()[0].upper() + termname.lower()[1:]
            queryWord2 = "_".join(termname2.split(" "))
            sparql = SPARQLWrapper("http://dbpedia.org/sparql")
            sparql.setQuery("""SELECT * WHERE {?synonyms <http://dbpedia.org/ontology/wikiPageRedirects> <http://dbpedia.org/resource/"""+ queryWord + """> .}""")
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
                res = result["synonyms"]["value"]
                name = res.split('/')[-1]
                tempList.append([name, termname, "Synonym"])
        
        appendingList = []
        for elem in tempList:
            if elem not in appendingList:
                appendingList.append(elem)
        final_list.extend(appendingList)
        
    return final_list

dbpedia_terms_unfiltered = dbpedia_parse(concepts)

string = "\n".join(list(set(" ".join("\t".join(a).lower().split("_")) for a in 
                            [l for l in dbpedia_hypernyms_unfiltered if l[0]!=l[1]])))
open("terms_from_dbpedia_" + domainName + ".tsv", "w+").write(string)
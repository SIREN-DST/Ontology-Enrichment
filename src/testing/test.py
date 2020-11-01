import pickle, sys, os, configparser
import numpy as np
from itertools import count
from collections import defaultdict
import tensorflow as tf
import tensorflow_hub as hub

config = configparser.ConfigParser()
try:
    config.read('config.ini')
except:
    print ("ERROR: No config file. Create a new file called config.ini")
    exit()

def check_field(section, key, key_name, optional=False):
	''' Checks config.ini for existence of config[section][key], and also whether
	 that refers to a file that exists. Prints a Warning or Error accordingly
	 Args:
	- section: Refers to a section in `config.ini`
	- key: Refers to a key under section in `config.ini`
	- key_name: Refers to the field being queried. Used to generate error msg if needed
	- optional: Signifies whether presence of key in `config.ini` is optional or not
	  '''
	try:
		field = os.path.abspath(config[section][key])
	except:
		if optional:
			print ("WARNING: No " + key_name + " specified in config.ini")
			return
		else:
		    raise KeyError("ERROR: No " + key_name + " specified. Check config.ini")

	if os.path.exists(field):
	    return field
	else:
		if optional:
			print ("WARNING: No file found by the name of", config[section][key])
			return
		else:
			raise FileNotFoundError("No file found by the name of", config[section][key])

# Domain of ontology. Used for naming purposes
domain = check_field(config['DEFAULT']['domain'], "domain name")

# Datasets
webpage_dir = check_field('dataset', 'webpages_dir', "Webpage directory")

# Filtering parameters
domain_keyword = check_field('filtering', 'domain_keyword', "Domain Keyword")
domain_threshold = float(check_field('filtering', 'domain_threshold', "Domain Threshold"))
inter_threshold = float(check_field('filtering', 'inter_threshold', "Inter Threshold"))

failed, success = [], []
relations = ["hypernym", "hyponym", "concept", "instance", "none"]
NUM_RELATIONS = len(relations)

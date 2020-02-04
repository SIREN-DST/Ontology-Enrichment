# README

## Project Structure

1. `dataset-creation`: Consists of code to create training data from the seed ontology and online sources like DBpedia, Wikidata and Wordnet. It consists of the following subdirectories:
		a. `files`: Various files created from the extraction of hypernyms and hyponyms from the online sources mentioned above
		b. `ontologies`: Security seed ontologies used for enrichment
		c. `wikiextractor` : Extract (and filter based on domain) articles in plain text from a wikipedia dump
2. `LSTM-implementation`: Consists of an LSTM trained on a wiki-dump corpus using the datasets created from scripts in  `dataset-creation`

## How to Run

To create the dataset:

1. Go to `dataset-creation`. 
2. To create training dataset, run `python3 extract_training_dataset.py`  to extract the untagged training corpus from online sources using the concepts in the seed ontologies.
3. To create testing dataset, substitute the `url` variable in `extract_testing_dataset.py`with the url you want to extract concepts and relationships from. `thresholdWord`is an optional parameter that can be used to filter relevant terms if the size of the testing dataset is too large.
4. Then, run `python3 extract_testing_dataset.py`  to extract testing dataset from the URL given above.

To train the LSTM model:

1. Run bash `create_resource_from_corpus.sh [wiki_dump_file] [resource_prefix]` where resource_prefix is the file path and prefix of the corpus files, e.g. security, such that the directory corpus will eventually contain the security_*.db files created by this script.
2. Run `python3 train_integrated.py [resource_prefix] [dataset_prefix] [model_prefix_file] [embeddings_file] [alpha] [word_dropout_rate]`
where:
	- `resource_prefix` is the file path and prefix of the corpus files, e.g. security, such that the directory corpus contains the security_*.db files created by create_resource_from_corpus.sh.
	- `dataset_prefix` is the file path of the dataset files, e.g. dataset/, such that this directory contains 3 files: train.tsv, test.tsv and val.tsv.
	- `model_prefix_file` is the output directory and prefix for the model files. The model is saved in 3 files: .model, .params and .dict. In addition, the test set predictions are saved in .predictions, and the prominent paths are saved to .paths.
	- `embeddings_file` is the pre-trained word embeddings file, in txt format (i.e., every line consists of the word, followed by a space, and its vector. We use [GloVe](http://nlp.stanford.edu/data/glove.6B.zip) to train our models.)
	- `alpha` is the learning rate (eg. 0.001).
	- `word_dropout_rate` is the word dropout rate. (eg. 0.3)


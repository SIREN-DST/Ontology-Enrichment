import sys
sys.argv.insert(1, '--dynet-mem')
sys.argv.insert(2, '16384')
sys.argv.insert(3, '--dynet-seed')
sys.argv.insert(4, '2840892268') # Change to any seed you'd like

from lstm import *
from itertools import count
from collections import defaultdict
from knowledge import KnowledgeResource
from pathLSTMClassifier import PathLSTMClassifier
from sklearn.metrics import precision_recall_fscore_support

EMBEDDINGS_DIM = 300


def main():
    """
    Trains the LSTM-based integrated pattern-based and distributional method for hypernymy detection
    :return:
    """

    corpus_prefix = sys.argv[5]
    dataset_prefix = sys.argv[6]
    output_file = sys.argv[7]
    embeddings_file = sys.argv[8]
    alpha = float(sys.argv[9])
    word_dropout_rate = float(sys.argv[10])

    np.random.seed(133)
    relations = ['False', 'True']

    # Load the datasets
    print('Loading the dataset...')
    train_set = load_dataset(dataset_prefix + 'train.tsv')
    test_set = load_dataset(dataset_prefix + 'test.tsv')
    val_set = load_dataset(dataset_prefix + 'val.tsv')
    y_train = [1 if 'True' in train_set[key] else 0 for key in list(train_set.keys())]
    y_test = [1 if 'True' in test_set[key] else 0 for key in list(test_set.keys())]
    # Uncomment if you'd like to load the validation set (e.g. to tune the hyper-parameters)
    # y_val = [1 if 'True' in val_set[key] else 0 for key in val_set.keys()]
    dataset_keys = list(train_set.keys()) + list(test_set.keys()) + list(val_set.keys())
    print('Done!')

    # Load the word embeddings
    print('Initializing word embeddings...')
    if embeddings_file is not None:
        wv, lemma_index = load_embeddings(embeddings_file)

    lemma_inverted_index = { i : w for w, i in lemma_index.items() }

    # Load the paths and create the feature vectors
    print('Loading path files...')
    x_y_vectors, dataset_instances, pos_index, dep_index, dir_index, \
    pos_inverted_index, dep_inverted_index, dir_inverted_index = load_paths(corpus_prefix, dataset_keys, lemma_index)
    print('Done!')
    print('Number of lemmas %d, number of pos tags: %d, number of dependency labels: %d, number of directions: %d' % \
          (len(lemma_index), len(pos_index), len(dep_index), len(dir_index)))

    X_train = dataset_instances[:len(train_set)]
    X_test = dataset_instances[len(train_set):len(train_set)+len(test_set)]
    # Uncomment if you'd like to load the validation set (e.g. to tune the hyper-parameters)
    # X_val = dataset_instances[len(train_set)+len(test_set):]

    x_y_vectors_train = x_y_vectors[:len(train_set)]
    x_y_vectors_test = x_y_vectors[len(train_set):len(train_set)+len(test_set)]
    # Uncomment if you'd like to load the validation set (e.g. to tune the hyper-parameters)
    # x_y_vectors_val = x_y_vectors[len(train_set)+len(test_set):]

    # Create the classifier
    classifier = PathLSTMClassifier(num_lemmas=len(lemma_index), num_pos=len(pos_index),
                                    num_dep=len(dep_index),num_directions=len(dir_index), n_epochs=3,
                                    num_relations=2, lemma_embeddings=wv, dropout=word_dropout_rate, alpha=alpha,
                                    use_xy_embeddings=True)

    # print 'Training with regularization = %f, learning rate = %f, dropout = %f...' % (reg, alpha, dropout)
    print('Training with learning rate = %f, dropout = %f...' % (alpha, word_dropout_rate))
    classifier.fit(X_train, y_train, x_y_vectors=x_y_vectors_train)

    print('Evaluation:')
    pred = classifier.predict(X_test, x_y_vectors=x_y_vectors_test)
    p, r, f1, support = precision_recall_fscore_support(y_test, pred, average='binary')
    print('Precision: %.3f, Recall: %.3f, F1: %.3f' % (p, r, f1))

    # Save the best model to a file
    classifier.save_model(output_file, [lemma_index, pos_index, dep_index, dir_index])

    # Write the predictions to a file
    output_predictions(output_file + '.predictions', relations, pred, list(test_set.keys()), y_test)

    # Retrieve k-best scoring paths
    all_paths = unique([path for path_list in dataset_instances for path in path_list])
    top_k = classifier.get_top_k_paths(all_paths, 1000)

    with codecs.open(output_file + '.paths', 'w', 'utf-8') as f_out:
        for path, score in top_k:
            path_str = '_'.join([reconstruct_edge(edge, lemma_inverted_index, pos_inverted_index,
                                                  dep_inverted_index, dir_inverted_index) for edge in path])
            print('\t'.join([path_str, str(score)]), file=f_out)


def load_paths(corpus_prefix, dataset_keys, lemma_index):
    """
    Override load_paths from lstm_common to include (x, y) vectors
    :param corpus_prefix:
    :param dataset_keys:
    :return:
    """

    # Define the dictionaries
    pos_index = defaultdict(count(0).__next__)
    dep_index = defaultdict(count(0).__next__)
    dir_index = defaultdict(count(0).__next__)

    dummy = pos_index['#UNKNOWN#']
    dummy = dep_index['#UNKNOWN#']
    dummy = dir_index['#UNKNOWN#']

    # Load the resource (processed corpus)
    print('Loading the corpus...')
    corpus = KnowledgeResource(corpus_prefix)
    print('Done!')

    keys = [(corpus.get_id_by_term(str(x)), corpus.get_id_by_term(str(y))) for (x, y) in dataset_keys]
    paths_x_to_y = [{ vectorize_path(path, lemma_index, pos_index, dep_index, dir_index) : count
                      for path, count in get_paths(corpus, x_id, y_id).items() }
                    for (x_id, y_id) in keys]
    paths_x_to_y = [ { p : c for p, c in paths_x_to_y[i].items() if p is not None } for i in range(len(keys)) ]

    paths = paths_x_to_y

    empty = [dataset_keys[i] for i, path_list in enumerate(paths) if len(list(path_list.keys())) == 0]
    print('Pairs without paths:', len(empty), ', all dataset:', len(dataset_keys))

    # Get the word embeddings for x and y (get a lemma index)
    x_y_vectors = [(lemma_index.get(x, 0), lemma_index.get(y, 0)) for (x, y) in dataset_keys]

    pos_inverted_index = { i : p for p, i in pos_index.items() }
    dep_inverted_index = { i : p for p, i in dep_index.items() }
    dir_inverted_index = { i : p for p, i in dir_index.items() }

    return x_y_vectors, paths, pos_index, dep_index, dir_index, \
           pos_inverted_index, dep_inverted_index, dir_inverted_index


if __name__ == '__main__':
    main()

import pickle
import tensorflow_hub as hub

USE_link = "https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed"
USE_model = hub.load(USE_link)
NULL_PATH = ((0, 0, 0, 0),)

arrow_heads = {">": "up", "<":"down"}


def id_to_entity(db, entity_id):
    ''' Lookup db for entity using ID '''
    entity = db[str(entity_id)]
    return entity

def id_to_path(db, entity_id):
    ''' Lookup db for path using ID '''
    entity = db[str(entity_id)]
    entity = "/".join(["*##*".join(e.split("_", 1)) for e in entity.split("/")])
    return entity

def entity_to_id(db, entity, resolve=True):
    ''' Lookup db for entity ID. In case of missing word, 
    if `resolve_db` is present, use it otherwise return None '''
    global success, failed
    entity_id = db.get(entity)
    if entity_id:
        success.append(entity)
        return int(entity_id)
    if not resolve:
        return -1
    closest_entity = resolved.get(entity, "")
    if closest_entity and closest_entity[0] and float(closest_entity[1]) > resolve_threshold:
        success.append(entity)
        return int(db[closest_entity[0]])
    failed.append(entity)
    return -1

def extract_paths(db, x, y):
    '''Extract paths between `x` and `y` from `db` and serialize it into a dictionary'''
    key = (str(x) + '###' + str(y))
    try:
        relation = db[key]
        return {int(path_count.split(":")[0]): int(path_count.split(":")[1]) for path_count in relation.split(",")}
    except Exception as e:
        return {}

def preprocess_db(db):
    '''Decodes db keys and values to utf-8'''
    final_db = {}
    for key in db:
        try:
            new_key = key.decode("utf-8")
        except:
            new_key = key
        try:
            new_val = db[key].decode("utf-8")
        except:
            new_val = db[key]
        final_db[new_key] = new_val
    return final_db

def load_db(db_name, encoded=True):
    ''' Loads pickle file. If `encoded`, it also decodes them. '''
    if not db_name:
        return
    return preprocess_db(pickle.load(open(db_name, "rb")))

def extractUSEEmbeddings(words):
    word_embeddings = USE_model(words)
    return word_embeddings.numpy()

def to_list_mixed(seq):
    '''Converts mixed list of tuples into list'''
    for item in seq:
        if isinstance(item, tuple):
            yield list(to_list_mixed(item))
        elif isinstance(item, list):
            yield [list(to_list_mixed(elem)) for elem in item]
        else:
            yield item

def extract_direction(edge):
    '''Converts direction arrow heads into string representation based on positions'''
    if edge[0] == ">" or edge[0] == "<":
        direction = "start_" + arrow_heads[edge[0]]
        edge = edge[1:]
    elif edge[-1] == ">" or edge[-1] == "<":
        direction = "end_" + arrow_heads[edge[-1]]
        edge = edge[:-1]
    else:
        direction = ' '
    return direction, edge

def parse_path(path):
    '''Parses a path by: 
    1. Serializing it by converting into a sequence of edges.
    2. Indexing word, POS, dependency and direction tags to represent edge as 4-tuple'''
    parsed_path = []
    for edge in path.split("*##*"):
        direction, edge = extract_direction(edge)
        if edge.split("/"):
            try:
                embedding, pos, dependency = tuple([a[::-1] for a in edge[::-1].split("/",2)][::-1])
            except:
                print (edge, path)
                raise
            emb_idx, pos_idx, dep_idx, dir_idx = emb_indexer[embedding], pos_indexer[pos], dep_indexer[dependency], dir_indexer[direction]
            parsed_path.append(tuple([emb_idx, pos_idx, dep_idx, dir_idx]))
        else:
            return None
    return tuple(parsed_path)

def parse_tuple(tup, resolve=True):
    '''Extracts paths between a pair of entities (both X->Y and Y->X)'''
    x, y = [entity_to_id(word2id_db, elem, resolve) for elem in tup]
    paths_x, paths_y = list(extract_paths(relations_db,x,y).items()), list(extract_paths(relations_db,y,x).items())
    path_count_dict_x = { id_to_path(id2path_db, path).replace("X/", tup[0]+"/").replace("Y/", tup[1]+"/") : freq for (path, freq) in paths_x }
    path_count_dict_y = { id_to_path(id2path_db, path).replace("Y/", tup[0]+"/").replace("X/", tup[1]+"/") : freq for (path, freq) in paths_y }
    path_count_dict = {**path_count_dict_x, **path_count_dict_y}
    return path_count_dict

def parse_dataset(dataset, resolve=True):
    '''Main function used to parse dataset. For every pair of entity, it returns
    a) the (serialized and indexed) paths between them 
    b) the count (or frequency of occurence) of each of these paths
    c) the target label'''
    parsed_dicts = [parse_tuple(tup, resolve) for tup in dataset.keys()]
    parsed_dicts = [{ parse_path(path) : path_count_dict[path] for path in path_count_dict } for path_count_dict in parsed_dicts]
    paths = [{ path : path_count_dict[path] for path in path_count_dict if path} for path_count_dict in parsed_dicts]
    paths = [{NULL_PATH: 1} if not path_list else path_list for i, path_list in enumerate(paths)]
    counts = [list(path_dict.values()) for path_dict in paths]
    paths = [list(path_dict.keys()) for path_dict in paths]
    targets = [rel_indexer[relation] for relation in dataset.values()]
    return list(to_list_mixed(paths)), counts, targets
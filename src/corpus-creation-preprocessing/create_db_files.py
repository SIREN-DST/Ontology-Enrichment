import sys, re, pickle

def processTerm(term):
    term = " ".join(re.sub("[^A-Za-z0-9 ]+", " ", term).strip().split())
    term = term.encode("ascii", "ignore")
    return term.strip()

def indexPathTerm(words_file):

    with open(paths_folder + "/filtered_paths", encoding="utf-8") as paths:
        
        filtered_paths = []
        for path in paths:
            filtered_paths.append(path.strip())
        filtered_paths = list(set(filtered_paths))


        path_to_id_dict = {filtered_paths[i]:i for i in range(len(filtered_paths))}
        id_to_path_dict = {i:filtered_paths[i] for i in range(len(filtered_paths))}

        with open(paths_folder + "/" + prefix + '_path_to_id_dict.pkl', 'wb') as handle:
            pickle.dump(path_to_id_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(paths_folder + "/" + prefix + '_id_to_path_dict.pkl', 'wb') as handle:
            pickle.dump(id_to_path_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(words_file, "r", encoding="utf-8") as terms:
        words = [processTerm(term) for term in terms]
        
        word_to_id_dict = {words[i]:i for i in range(len(words))}
        id_to_word_dict = {i:words[i] for i in range(len(words))}

        with open(paths_folder + "/" + prefix + '_word_to_id_dict.pkl', 'wb') as handle:
            pickle.dump(word_to_id_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(paths_folder + "/" + prefix + '_id_to_word_dict.pkl', 'wb') as handle:
            pickle.dump(id_to_word_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)        


def getTripletIDFromDB(parsed_file):

    x = file.split("_")[-3]
    output_parsed = open(paths_folder + '/triplet_id_' + x, 'w+')

    with open(paths_folder + "/" + prefix + '_word_to_id_dict.pkl', 'rb') as handle:
        word_to_id_dict = pickle.load(handle)
    with open(paths_folder + "/" + prefix + '_path_to_id_dict.pkl', 'rb') as handle:
        path_to_id_dict = pickle.load(handle)

    with open(parsed_file) as parsed_inp:
        for line in parsed_inp:
            if line.strip():
                try:
                    x, y, path = line.strip().split('\t')
                except:
                    print ("Bytes?", line)
                    raise
            else:
                continue

            x, y = processTerm(x.strip()), processTerm(y.strip())
            
            try:
                x_id, y_id = word_to_id_dict[x], word_to_id_dict[y]
            except Exception as e:
                print (e, x, y)
                continue
            path_id = path_to_id_dict.get(path.strip(), -1)
            if path_id != -1:
                triplet = "\t".join((str(x_id), str(y_id), str(path_id)))
                output_parsed.write(triplet + "\n")

    output_parsed.close()

def indexWordPairs(parsed_file):

    word_occurence_map = {}

    with open(parsed_file) as inp:
        prev_line, prev_count = '', 0
        for line in inp:
            curr_line = "\t".join(line.strip().split("\t")[:-1])
            if curr_line == prev_line or (not prev_line):
                prev_count += int(line.strip().split("\t")[-1])
            else:
                x, y, path = prev_line.split('\t')
                key = str(x) + '_' + str(y)
                current = path + ":" + str(prev_count)
                if key in word_occurence_map:
                    pastkeys = word_occurence_map[key]
                    current =  pastkeys + ',' + current
                word_occurence_map[key] = current
                prev_count = 1
            prev_line = curr_line
        x, y, path = prev_line.split('\t')
        key = str(x) + '_' + str(y)
        current = path + ":" + str(prev_count)
        if key in word_occurence_map:
            pastkeys = word_occurence_map[key]
            current =  pastkeys + ',' + current
        word_occurence_map[key] = current
    
    with open(paths_folder + "/" + prefix + '_word_occurence_map.pkl', 'wb') as handle:
        pickle.dump(word_occurence_map, handle, protocol=pickle.HIGHEST_PROTOCOL)


paths_folder = sys.argv[1]
file = sys.argv[2]
prefix = sys.argv[3]
mode = sys.argv[4]
if mode=="1":
    indexPathTerm(file)
elif mode=="2":
    getTripletIDFromDB(file)
elif mode=="3":
    indexWordPairs(file)

    

import bsddb3


class KnowledgeResource:
    """
    Holds the resource graph data
    """
    def __init__(self, resource_prefix):
        """
        Init the knowledge resource
        :param resource_prefix - the resource directory and file prefix
        """
        self.term_to_id = bsddb3.btopen(resource_prefix + '_term_to_id.db', 'r')
        self.id_to_term = bsddb3.btopen(resource_prefix + '_id_to_term.db', 'r')
        self.path_to_id = bsddb3.btopen(resource_prefix + '_path_to_id.db', 'r')
        self.id_to_path = bsddb3.btopen(resource_prefix + '_id_to_path.db', 'r')
        self.l2r_edges = bsddb3.btopen(resource_prefix + '_l2r.db', 'r')

    def get_term_by_id(self, id):
        return self.id_to_term[str(id).encode("utf-8")]

    def get_path_by_id(self, id):
        return self.id_to_path[str(id).encode("utf-8")]

    def get_id_by_term(self, term):
        return int(self.term_to_id[term.encode("utf-8")]) if self.term_to_id.has_key(term.encode("utf-8")) else -1

    def get_id_by_path(self, path):
        return int(self.path_to_id[path.encode("utf-8")]) if self.path_to_id.has_key(path.encode("utf-8")) else -1

    def get_relations(self, x, y):
        """
        Returns the relations from x to y
        """
        path_dict = {}
        key = str(x) + '###' + str(y)
        path_str = self.l2r_edges[key.encode("utf-8")] if self.l2r_edges.has_key(key.encode("utf-8")) else ''

        if len(path_str) > 0:
            paths = [tuple(map(int, p.split(':'))) for p in path_str.split(',')]
            path_dict = { path : count for (path, count) in paths }

        return path_dict

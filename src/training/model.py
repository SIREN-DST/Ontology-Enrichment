import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

relations = ["hypernym", "hyponym", "concept", "instance", "none"]
NUM_RELATIONS = len(relations)
POS_DIM = 4
DEP_DIM = 6
DIR_DIM = 3

class OntoEnricher(nn.Module):

    def __init__(self, emb_vals):
        
        super(OntoEnricher, self).__init__()

        self.EMBEDDING_DIM = np.array(emb_vals).shape[1]
        self.n_directions = 2

        self.input_dim = POS_DIM + DEP_DIM + self.EMBEDDING_DIM + DIR_DIM
        self.output_dim = self.n_directions * HIDDEN_DIM * NUM_LAYERS + 2 * self.EMBEDDING_DIM

        self.layer1_dim = LAYER1_DIM
        self.W1 = nn.Linear(self.output_dim, self.layer1_dim)
        self.W2 = nn.Linear(self.layer1_dim, NUM_RELATIONS)

        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.hidden_dropout = nn.Dropout(p=hidden_dropout)
        self.output_dropout = nn.Dropout(p=output_dropout)
        self.log_softmax = nn.LogSoftmax()
        
        self.name_embeddings = nn.Embedding(len(emb_vals), self.EMBEDDING_DIM)
        self.name_embeddings.load_state_dict({'weight': torch.from_numpy(np.array(emb_vals))})
        self.name_embeddings.weight.requires_grad = False

        self.pos_embeddings = nn.Embedding(len(pos_indexer), POS_DIM)
        self.dep_embeddings = nn.Embedding(len(dep_indexer), DEP_DIM)
        self.dir_embeddings = nn.Embedding(len(dir_indexer), DIR_DIM)

        nn.init.xavier_uniform_(self.pos_embeddings.weight)
        nn.init.xavier_uniform_(self.dep_embeddings.weight)
        nn.init.xavier_uniform_(self.dir_embeddings.weight)
        
        self.lstm = nn.LSTM(self.input_dim, HIDDEN_DIM, NUM_LAYERS, bidirectional=True, batch_first=True)

    def masked_softmax(self, inp):
        # To softmax all non-zero tensor values
        inp = inp.double()
        mask = ((inp != 0).double() - 1) * 9999  # for -inf
        return (inp + mask).softmax(dim=-1)

    def forward(self, nodes, paths, counts, edgecounts, max_paths, max_edges):
        '''
            nodes: batch_size * 2
            paths: batch_size * max_paths * max_edges * 4
            counts: batch_size * max_paths
            edgecounts: batch_size * max_paths
        '''
        word_embed = self.emb_dropout(self.name_embeddings(paths[:,:,:,0]))
        pos_embed = self.emb_dropout(self.pos_embeddings(paths[:,:,:,1]))
        dep_embed = self.emb_dropout(self.dep_embeddings(paths[:,:,:,2]))
        dir_embed = self.emb_dropout(self.dir_embeddings(paths[:,:,:,3]))
        paths_embed = torch.cat((word_embed, pos_embed, dep_embed, dir_embed), dim=-1)
        nodes_embed = self.emb_dropout(self.name_embeddings(nodes)).reshape(-1, 2*self.EMBEDDING_DIM)

        paths_embed = paths_embed.reshape((-1, max_edges, self.input_dim))

        paths_packed = pack_padded_sequence(paths_embed, torch.flatten(edgecounts), batch_first=True, enforce_sorted=False)
        _, (hidden_state, _) = self.lstm(paths_packed)
        paths_output = self.hidden_dropout(hidden_state).permute(1,2,0)
        paths_output_reshaped = paths_output.reshape(-1, max_paths, HIDDEN_DIM*NUM_LAYERS*self.n_directions)
        # paths_output has dim (batch_size, max_paths, HIDDEN_DIM, NUM_LAYERS*self.n_directions)

        paths_weighted = torch.bmm(paths_output_reshaped.permute(0,2,1), counts.unsqueeze(-1)).squeeze(-1)
        representation = torch.cat((nodes_embed, paths_weighted), dim=-1)
        
        probabilities = self.log_softmax(self.W2(F.relu(self.W1(representation))))
        return probabilities
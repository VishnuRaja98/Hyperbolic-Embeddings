# import os
# import numpy as np
# import scipy.sparse.csgraph as csg
# from joblib import Parallel, delayed
# import multiprocessing
import networkx as nx
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from __future__ import unicode_literals, print_function, division
from io import open
# import unicodedata
# import string
# import re
# import random

#BEGIN imports ive added
import csv
# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from graphviz import Digraph
from sklearn.metrics.pairwise import cosine_similarity  # for nlp tasks cosine similarity measures closeness better than eucledian
import trainer
#END imports ive added

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_lines(path):
    """read data from tab seperated lines into list """
    lines = []
    with open(path, 'r', newline='', encoding='utf-8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for row in reader:
            lines.append(row)
    if VERBOSE: print(f"Total relations = {len(lines)}")
    return lines

class InputData:
    """stores all mapping, embedding data"""
    def __init__(self, lines, neg_lines) -> None:
        self.lines = lines
        self.neg_lines = neg_lines
        self.sentence_ids = {}
        self.unique_sentences_list = []
        self.eucledian_embeddings_all = []

        self.trees_dict = {}     # create an dict trees in the format {parent:[children]}
        self.neg_trees_dict = {}
        self.trees_list = []     # create an array of tree edges of the format [[parent,children]] like in the .edges files in ParseTreeEmbeddings.ipynb
        self.connected_components = []
        self.belongs_to_tree = {}
        self.trees = []
        self.euclidean_embeddings = []
        self.indices = []  # each connected_component index. Used in FcIters
    

    def sentence_id_mappings(self):
        """Initializes the mapping of sentence to ID and ID to sentence 
        sentence_id[sentence]=id
        unique_sentences_list[id]=sentence"""
        i=0
        for line in self.lines:
            if line[0] not in self.sentence_ids:
                self.sentence_ids[line[0]]=i
                i=i+1
            if line[1] not in self.sentence_ids:
                self.sentence_ids[line[1]]=i
                i=i+1
        
        for line in self.neg_lines:
            if line[0] not in self.sentence_ids:
                self.sentence_ids[line[0]]=i
                i=i+1
            if line[1] not in self.sentence_ids:
                self.sentence_ids[line[1]]=i
                i=i+1

        # if VERBOSE: print("Unique Sentences = ", self.sentence_ids)
        self.unique_sentences_list = ["" for _ in range(len(self.sentence_ids))]
        for key, val in self.sentence_ids.items():
            self.unique_sentences_list[val]=key
        if VERBOSE: print("unique_sentences_list length = ", len(self.unique_sentences_list))

    def get_sentence_embeddings(self, model):
        """Creates sentence embeddings and create final input matrix of embeddings of sentences of each tree
        Final main output = eucledian_embeddings = [[sentenceEmbedding1,..,sentenceEmbeddingN] for each distinct tree]"""
        self.eucledian_embeddings_all = torch.tensor(model.encode(self.unique_sentences_list))
        if VERBOSE: 
            print("embeddings shape = ",self.eucledian_embeddings_all.shape)
            # print("First sentence tokenized = ",self.eucledian_embeddings_all[0])
        # create an dict trees in the format {parent:[children]}
        # create an array of tree edges of the format [[parent,children]] like in the .edges files in ParseTreeEmbeddings.ipynb
        G = nx.Graph()
        for line in self.lines:
            parent_id = self.sentence_ids[line[0]]
            child_id = self.sentence_ids[line[1]]
            self.trees_list.append([parent_id,child_id])
            if parent_id in self.trees_dict:
                if child_id not in self.trees_dict[parent_id]:
                    self.trees_dict[parent_id].append(child_id)
            else:
                self.trees_dict[parent_id]=[child_id]

        # not adding the negative lines into tree edges cause dont know what to do with negative realtions right now
        # im just generating the negative trees, dont know if theyre useful
        for line in self.neg_lines:
            parent_id = self.sentence_ids[line[0]]
            child_id = self.sentence_ids[line[1]]
            if parent_id in self.neg_trees_dict:
                if child_id not in self.neg_trees_dict[parent_id]:
                    self.neg_trees_dict[parent_id].append(child_id)
            else:
                self.neg_trees_dict[parent_id]=[child_id]

        # finding connected components using trees_list
        G.add_edges_from(self.trees_list)
        self.connected_components = list(nx.connected_components(G))

        self.indices = [i for i in range(len(self.connected_components))]   # index for each Connected Component

        # now ill combine the trees_list data and connected_components data to form an array where each element will be a connected graph
        for i,tree in enumerate(self.connected_components):
            for node in tree:
                self.belongs_to_tree[node]=i
        self.trees = [[] for _ in range(len(self.connected_components))]
        for edge in self.trees_list:
            self.trees[self.belongs_to_tree[edge[0]]].append(edge)
        self.euclidean_embeddings = [torch.stack([self.eucledian_embeddings_all[i] for i in component]) for component in self.connected_components]
        # this final eucledian embeddings consists of list of list of eucledian embeddingd of each sentence of each tree
        if VERBOSE:
            # print("trees list = ", self.trees_list)
            # print("trees dict = ", self.trees_dict)  #tree_dict and neg_trees_dict is only for plotting the graph. Wont include them finally
            # print("negtive trees dict = ", self.neg_trees_dict)
            # print("Connected components = ", self.connected_components)
            print("Unique connected components = ", len(self.indices))
            # print("belongs_to_tree = ", self.belongs_to_tree)
            # print("trees = ", self.trees)
            print(f"euclidean_embeddings[0].shape={self.euclidean_embeddings[0].shape}")

    def draw_trees(self):
        # Create a sample tree structure
        tree_structure = self.trees_dict

        # Create a directed graph
        dot = Digraph(comment='Tree')

        # Add nodes and edges to the graph
        for parent, children in tree_structure.items():
            dot.node('"{}"'.format(self.unique_sentences_list[parent]))
            for child in children:
                dot.edge('"{}"'.format(self.unique_sentences_list[parent]), '"{}"'.format(self.unique_sentences_list[child]), \
                        label = str(cosine_similarity([self.eucledian_embeddings_all[parent]], [self.eucledian_embeddings_all[child]])))
                
        dot.render('tree_cosine', format='pdf', cleanup=True)

        # Adding connections to next unconnected tree to show the bert encoding similarities between some unrealted texts
        for i in range(len(self.connected_components)-1):
            first_tree_node = None
            for component in self.connected_components[i]:
                if component not in self.trees_dict:
                    first_tree_node = component
                    break
            second_tree_node = None
            for component in self.connected_components[i+1]:
                if component not in self.trees_dict:
                    second_tree_node = component
                    break

            dot.node('"{}"'.format(self.unique_sentences_list[first_tree_node]))
            dot.edge('"{}"'.format(self.unique_sentences_list[first_tree_node]), '"{}"'.format(self.unique_sentences_list[second_tree_node]), \
                        label = str(cosine_similarity([self.eucledian_embeddings_all[first_tree_node]], [self.eucledian_embeddings_all[second_tree_node]])), \
                        color="red")

        # Save the graph as a PDF or PNG
        dot.render('tree_cosine_with_unrealted_connection', format='pdf', cleanup=True)


# Define a neural network
class EucToHypNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(EucToHypNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 2*input_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2*input_size, output_size)
        self.softmax_out = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax_out(x)
        return x
# another option is to directly finetune the sentence embeding model with hyperbolic loss functions
    
def main(): 
    global VERBOSE
    VERBOSE = True

    file_path_pos = 'pos_headline_pairs.tsv'
    file_path_neg = 'neg_headline_pairs.tsv'
    lines = read_lines(file_path_pos)   # positive lines
    neg_lines = read_lines(file_path_neg)   # positive lines

    data = InputData(lines, neg_lines)
    data.sentence_id_mappings()

    # Initilaly get Eucledian embeddings using this model. We will later transform these sentence embeddings into hyperbolic space
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    data.get_sentence_embeddings(model)



    input_size = 384
    output_size = 50

    converterModel = EucToHypNN(input_size, output_size)
    """mapping = nn.Sequential(
            nn.Linear(input_size, 384).to(device),
            nn.ReLU().to(device),
            nn.Linear(384, output_size).to(device),
            nn.ReLU().to(device)
            )"""
    
    # print("trees list = ", trees_list)
    # print("Connected components = ", connected_components)
    # print("Unique connected components = ", len(connected_components))
    # print("All trees = ", trees)
    # print("total trees = ", len(trees ))
    
    # trainer.trainFCIters(data, mapping)
    trainer.trainFCIters2(data, converterModel)

    # new_embeddings = mapping(torch.tensor(data.eucledian_embeddings_all))
    new_embeddings = converterModel(torch.tensor(data.eucledian_embeddings_all))
    print("New embeddings = ",new_embeddings)



if __name__=="__main__": 
    main() 


# Code from learning_utils.py
# EPS = 1e-15 # for numerical stability
# PROJ_EPS = 1e-5
# MAX_TANH_ARG = 15.0

# Dot prod
# def torch_dot(x, y):
#     return torch.sum(x * y, dim=1, keepdim=True)

# # L2 norm
# def torch_norm(x):
#     return torch.norm(x, dim=1, keepdim=True)

# # Inverse hyperbolic tangent applied element-wise to a tensor
# def t_arctanh(v):
#     return 0.5*torch.log((1+v)/(1-v))

# # Inverse hyperbolic tangent with clipping
# def torch_atanh(x):
#     return t_arctanh(torch.min(x, torch.tensor(1. - EPS)))  # just clips values and passes to arctanh which does the actual log(1+v/1-v)

# # Hyperbolic tangent with clipping
# def torch_tanh(x):
#     return torch.tanh(torch.min(torch.max(x, torch.tensor(-MAX_TANH_ARG)), torch.tensor(MAX_TANH_ARG)))

# # its fully commented out??
# def torch_project_hyp_vec(v, c=1):
#     """Projects the hyperbolic vectors to the inside of the ball."""
#     # clip_norm = torch.tensor(1-PROJ_EPS)
#     # clipped_v = F.normalize(v, p=2, dim=1)*clip_norm
#     # return clipped_v
#     return v

# def torch_hyp_add(u, v, c=1):
#     """Accepts torch tensors u, v and returns their sum in hyperbolic
#     space in tensor format. Radius of the open ball is 1/sqrt(c). """
#     v = v+torch.tensor(EPS)
#     torch_dot_u_v = 2 * torch_dot(u, v)
#     torch_norm_u_sq = torch_dot(u,u)
#     torch_norm_v_sq = torch_dot(v,v)
#     denominator = 1. + torch_dot_u_v + torch_norm_v_sq * torch_norm_u_sq
#     result = (1. + torch_dot_u_v + torch_norm_v_sq) / denominator * u + (1. - torch_norm_u_sq) / denominator * v
#     return torch_project_hyp_vec(result)

# def torch_mv_mul_hyp(M, x, c=1):
#     x = x + torch.tensor(EPS)
#     Mx = torch.matmul(x, M)+torch.tensor(EPS)
#     MX_norm = torch_norm(Mx)
#     x_norm = torch_norm(x)
#     result = torch_tanh(MX_norm / x_norm * torch_atanh(x_norm)) / MX_norm * Mx
#     return torch_project_hyp_vec(result, c)

# #Hyperbolic modules.

# class HypLinear(nn.Module):
#     """Applies a hyperbolic "linear" transformation to the incoming data: :math:`y = xA^T + b`
#        Uses hyperbolic formulation of addition, scaling and matrix multiplication.
#     """

#     def __init__(self, in_features, out_features, bias=True):
#         super(HypLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))

#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(1, out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, input_):
#         result = torch_hyp_add(torch_mv_mul_hyp(torch.transpose(self.weight,0,1), input_), self.bias) #(batch, input) x (input, output)
#         return result

#     def extra_repr(self):
#         return 'in_features={}, out_features={}, bias={}'.format(
#             self.in_features, self.out_features, self.bias is not None
#         )




# import os
import numpy as np
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
import matplotlib.pyplot as plt
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
    # if VERBOSE: print(f"Total relations = {len(lines)}")
    return lines

def plot_losses(losses):
    """
    Plots losses over iterations.
    losses (list): List of loss values over iterations.
    """
    plt.plot(losses, label='Loss')
    plt.title('Loss over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("train_losses.png")

def find_reachable_nodes(graph, start_node):
    """finds all super children for parents"""
    # Use breadth-first search to find reachable nodes
    reachable_nodes = set(nx.descendants(graph, start_node))
    
    # Include the start node itself
    reachable_nodes.add(start_node)
    
    return reachable_nodes

def add_edges_to_reachable_nodes(graph):
    """Adds a edge of weight 1 from a parent to its every super child"""
    # Create a new graph to store the result
    graph_with_edges = graph.copy()
    
    # Iterate through each node in the graph
    for node in graph.nodes():
        # Find reachable nodes from the current node
        reachable_nodes = find_reachable_nodes(graph, node)
        
        # Add edges from the current node to its reachable nodes
        for reachable_node in reachable_nodes:
            if node != reachable_node:  # Avoid self-loops
                graph_with_edges.add_edge(node, reachable_node)
    
    return graph_with_edges

def add_connections_with_common_parent(graph):
    """This created a connection of weight 2 if two nodes were on seperate brancehes of a common parent"""
    # Create a new graph to store the result
    graph_with_connections = graph.copy()
    
    # Iterate through each parent node in the graph
    for parent in graph.nodes():
        # Find children of the current parent
        children = list(graph.successors(parent))
        num_children = len(children)
        
        # Add connections with weight 2 between pairs of children
        for i in range(num_children):
            for j in range(num_children):
                child1 = children[i]
                child2 = children[j]
                # Check if there is no path between child1 and child2
                if not nx.has_path(graph, child1, child2) and not nx.has_path(graph, child2, child1):
                    graph_with_connections.add_edge(child1, child2, weight=2)
    
    return graph_with_connections

class InputData:
    """stores all mapping, embedding data"""
    def __init__(self, lines) -> None:
        self.lines = lines
        self.sentence_ids = {}
        self.unique_sentences_list = []
        self.eucledian_embeddings_all = []

        self.trees_dict = {}     # create an dict trees in the format {parent:[children]}
        self.trees_list = []     # create an array of tree edges of the format [[parent,children]] like in the .edges files in ParseTreeEmbeddings.ipynb
        self.connected_components = []
        self.belongs_to_tree = {}
        self.trees = []
        self.euclidean_embeddings = []
        self.indices = []  # each connected_component index. Used in FcIters
    
    def send_to_gpu(self):
        if torch.cuda.is_available():
            self.eucledian_embeddings_all = self.eucledian_embeddings_all.to(device)
            self.unique_sentences_list = self.unique_sentences_list.to(device)
            print("Variables sent to GPU.")
        else:
            print("GPU not available.")

    def to_cpu(self):
        self.eucledian_embeddings_all = self.eucledian_embeddings_all.cpu()
        self.unique_sentences_list = self.unique_sentences_list.cpu()
        print("Variables moved to CPU.")

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
        

        # if VERBOSE: print("Unique Sentences = ", self.sentence_ids)
        self.unique_sentences_list = ["" for _ in range(len(self.sentence_ids))]
        for key, val in self.sentence_ids.items():
            self.unique_sentences_list[val]=key
        # if VERBOSE: print("unique_sentences_list length = ", len(self.unique_sentences_list))

    def get_sentence_embeddings(self, model=None, pretrained_file = None):
        """Creates sentence embeddings and create final input matrix of embeddings of sentences of each tree
        Final main output = eucledian_embeddings = [[sentenceEmbedding1,..,sentenceEmbeddingN] for each distinct tree]"""
        if pretrained_file: # these will load embeddings from pretrained embeddings. There are not eucleding right now
            # VERBOSE = True
            self.eucledian_embeddings_all = torch.load("xlmr-embeddings/new_embeddings.pt", map_location=torch.device('cpu'))
        elif model:
            self.eucledian_embeddings_all = torch.tensor(model.encode(self.unique_sentences_list))
        else:
            raise ValueError("Need to pass atleast one of model or pretrained file to InputData.get_sentence_embeddings()")
        if VERBOSE: 
            print("embeddings shape = ",self.eucledian_embeddings_all.shape)
            # print("First sentence tokenized = ",self.eucledian_embeddings_all[0])
        # create an dict trees in the format {parent:[children]}
        # create an array of tree edges of the format [[parent,children]] like in the .edges files in ParseTreeEmbeddings.ipynb
        G = nx.Graph()
        G_directed = nx.DiGraph()   # First creating a directed graph to control the parent child relation ship
        # later once functions are completed, well convert to undirected if required
        for line in self.lines:
            parent_id = self.sentence_ids[line[0]]
            child_id = self.sentence_ids[line[1]]
            self.trees_list.append([parent_id,child_id])
            
        # print("trees_list", self.trees_list)
        G_directed.add_edges_from(self.trees_list)

        # Add edges from every node to each of its reachable nodes
        graph_with_edges = add_edges_to_reachable_nodes(G_directed) # this needs to also update trees_list as its whats being used in trainer.distance_hyperbolic
        self.trees_list = list(graph_with_edges.edges())
        # this below for loop is only used for creating datastructure for the pdf
        for edge in graph_with_edges.edges(data=True):
            # using this directed graph for drawing as the directed one wasnt very clear.
            parent_id = edge[0]
            child_id = edge[1]
            if parent_id in self.trees_dict:
                if child_id not in self.trees_dict[parent_id]:
                    self.trees_dict[parent_id].append(child_id)
            else:
                self.trees_dict[parent_id]=[child_id]

        # # Add connections with weight 2 between nodes that have a common parent
        # graph_with_connections = add_connections_with_common_parent(graph_with_edges)
        
        # converting to undirected to be able to calclate distances using djikstra later
        G = graph_with_edges.to_undirected()  
        
        # print("Edges in the graph:")
        # for edge in G.edges(data=True):            
        #     print(edge)
        
        # # maybe can include this function from trainer here itself if more efficient!
        # lengths = dict(nx.all_pairs_dijkstra_path_length(G))
        # # print("lengths = ", lengths)
        # n = nx.number_of_nodes(G)
        # dist_mat = np.full((n, n), np.inf)

        # for i, (node, lengths_to_node) in enumerate(lengths.items()):
        #     for target_node, length in lengths_to_node.items():
        #         dist_mat[node,target_node] = length

        # print(dist_mat)


        # finding connected components using trees_list
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

    def draw_trees(self, figname, sentence_embeddings = None, distance_metric=None):
        if sentence_embeddings is None:
            sentence_embeddings = self.eucledian_embeddings_all
        # Create a sample tree structure
        tree_structure = self.trees_dict

        # Create a directed graph
        dot = Digraph(comment='Tree')

        # Add nodes and edges to the graph
        for parent, children in tree_structure.items():
            dot.node('"{}"'.format(self.unique_sentences_list[parent]))
            for child in children:
                if not distance_metric:
                    dot.edge('"{}"'.format(self.unique_sentences_list[parent]), '"{}"'.format(self.unique_sentences_list[child]), \
                            label = str(cosine_similarity([sentence_embeddings[parent]], [sentence_embeddings[child]])))
                else:
                    dot.edge('"{}"'.format(self.unique_sentences_list[parent]), '"{}"'.format(self.unique_sentences_list[child]), \
                            label = str(distance_metric(sentence_embeddings[parent], sentence_embeddings[child]).item()))
                
        dot.render(figname, format='pdf', cleanup=True)

        # # Adding connections to next unconnected tree to show the bert encoding similarities between some unrealted texts
        # for i in range(len(self.connected_components)-1):
        #     first_tree_node = None
        #     for component in self.connected_components[i]:
        #         if component not in self.trees_dict:
        #             first_tree_node = component
        #             break
        #     second_tree_node = None
        #     for component in self.connected_components[i+1]:
        #         if component not in self.trees_dict:
        #             second_tree_node = component
        #             break

        #     dot.node('"{}"'.format(self.unique_sentences_list[first_tree_node]))
        #     dot.edge('"{}"'.format(self.unique_sentences_list[first_tree_node]), '"{}"'.format(self.unique_sentences_list[second_tree_node]), \
        #                 label = str(cosine_similarity([self.eucledian_embeddings_all[first_tree_node]], [self.eucledian_embeddings_all[second_tree_node]])), \
        #                 color="red")

        # # Save the graph as a PDF or PNG
        # dot.render('tree_cosine_with_unrealted_connection', format='pdf', cleanup=True)


# Define a neural network
class EucToHypNN_old(nn.Module):
    def __init__(self, input_size, output_size):
        super(EucToHypNN, self).__init__()
        self.l1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
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
    
class EucToHypNN(nn.Module):
    def __init__(self, output_size, sentence_transformer, freeze_l1=False):
        super(EucToHypNN, self).__init__()
        # self.l1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.l1 = sentence_transformer

        if freeze_l1:
            for param in self.l1.parameters():
                param.requires_grad = False

        self.fc1 = nn.Linear(self.l1.get_sentence_embedding_dimension(), 2*self.l1.get_sentence_embedding_dimension())
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2*self.l1.get_sentence_embedding_dimension(), output_size)
        # self.softmax_out = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.tensor(self.l1.encode(x)).to(device)
        # Normalize the embeddings -> didnt make a diff
        # x = torch.nn.functional.normalize(x, p=2, dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.softmax_out(x)
        return x
# another option is to directly finetune the sentence embeding model with hyperbolic loss functions

def main(): 
    global VERBOSE
    VERBOSE = True

    file_path_pos = 'pos_headline_pairs.tsv'
    # file_path_neg = 'neg_headline_pairs.tsv'
    lines = read_lines(file_path_pos)   # positive lines
    # neg_lines = read_lines(file_path_neg)   # positive lines

    data = InputData(lines)
    data.sentence_id_mappings()

    sentence_transformer_name = 'sentence-transformers/stsb-xlm-r-multilingual' # 'sentence-transformers/all-MiniLM-L6-v2'
    # Initilaly get Eucledian embeddings using this model. We will later transform these sentence embeddings into hyperbolic space
    model = SentenceTransformer(sentence_transformer_name)
    data.get_sentence_embeddings(model)

    data.draw_trees(figname = "tree_cosine")   # optional
    # data.draw_trees(figname = "tree_hyp_initial", distance_metric = trainer.dist_h)   # optional

    output_size = 768   # 384

    if device == 'cuda':
        data.send_to_gpu()

    converterModel = EucToHypNN(output_size,model)
    print("ConverterModel = ", converterModel)
    converterModel.to(device)

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
    epochs = 100
    train_losses = trainer.trainFCIters2(data, converterModel, n_epochs=epochs)
    torch.save(converterModel.state_dict(), 'converterModel.pt')
    plot_losses(train_losses)

    # print("data.eucledian_embeddings_all = ", data.eucledian_embeddings_all)
    # new_embeddings = mapping(torch.tensor(data.eucledian_embeddings_all))
    new_embeddings = converterModel(data.unique_sentences_list)

    # print("old = ",data.eucledian_embeddings_all[0])
    # print("new = ",new_embeddings[0])

    torch.save(new_embeddings, 'new_embeddings.pt')
    
    data.draw_trees(figname = "tree_hyp_final", sentence_embeddings=new_embeddings.detach(), distance_metric = trainer.dist_h)   # optional



if __name__=="__main__": 
    main() 






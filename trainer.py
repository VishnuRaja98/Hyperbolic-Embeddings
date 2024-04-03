import torch
import math
import time
import networkx as nx
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Riemannian SGD

from torch.optim.optimizer import Optimizer, required
spten_t = torch.sparse.FloatTensor


def poincare_grad(p, d_p):
    """
    Calculates Riemannian grad from Euclidean grad.
    Args:
        p (Tensor): Current point in the ball
        d_p (Tensor): Euclidean gradient at p
    """
    if d_p.is_sparse:
        p_sqnorm = torch.sum(
            p.data[d_p._indices()[0].squeeze()] ** 2, dim=1,
            keepdim=True
        ).expand_as(d_p._values())
        n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
        d_p = spten_t(d_p._indices(), n_vals, d_p.size())
    else:
        p_sqnorm = torch.sum(p.data ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)

    return d_p


def euclidean_grad(p, d_p):
    return d_p


def retraction(p, d_p, lr):
    # Gradient clipping.
    if torch.all(d_p < 1000) and torch.all(d_p>-1000):
        p.data.add_(-lr, d_p)


class RiemannianSGD(Optimizer):
    """Riemannian stochastic gradient descent.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rgrad (Function): Function to compute the Riemannian gradient from
            an Euclidean gradient
        retraction (Function): Function to update the parameters via a
            retraction of the Riemannian gradient
        lr (float): learning rate
    """

    def __init__(self, params, lr=required, rgrad=required, retraction=required):
        defaults = dict(lr=lr, rgrad=rgrad, retraction=retraction)
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None):
        """Performs a single optimization step.
        Arguments:
            lr (float, optional): learning rate for the current update.
        """
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if lr is None:
                    lr = group['lr']
                d_p = group['rgrad'](p, d_p)
                group['retraction'](p, d_p, lr)

        return loss
    
# Distortion calculations

def acosh(x):   # acosh = log(x+sqrt(x^2 - 1))
    return torch.log(x + torch.sqrt(x**2-1))

def dist_h(u,v):
    z  = 2*torch.norm(u-v,2)**2 # formula at - https://en.wikipedia.org/wiki/Poincar%C3%A9_disk_model
    uu = 1. + torch.div(z,((1-torch.norm(u,2)**2)*(1-torch.norm(v,2)**2)))  # we can just just use torch.norm(u) instead of torch.norm(u,2) L2 is by default
    return acosh(uu)

def distance_matrix_hyperbolic(input):
    row_n = input.shape[0]
    dist_mat = torch.zeros(row_n, row_n, device=device) # dist_mat stores the hyperbolic (geodesic) distance between each pair of points
    for row in range(row_n):
        for i in range(row_n):
            if i != row:
                dist_mat[row, i] = dist_h(input[row,:], input[i,:])
    return dist_mat

# test_input = torch.tensor([[1,1,1,1,1,1,1,1],[0,3,1,0,0,3,1,0],[1,0,0,1,1,0,0,1], [2,0,0,2,2,0,0,2]], dtype=torch.float32)
# print("test_input = ", test_input)
# print(distance_matrix_hyperbolic(test_input))

def entry_is_good(h, h_rec): return (not torch.isnan(h_rec)) and (not torch.isinf(h_rec)) and h_rec != 0 and h != 0

def distortion_entry(h,h_rec):
    avg = abs(h_rec - h)/h
    return avg

def distortion_row(H1, H2, n, row):
    avg, good = 0, 0
    for i in range(n):
        if i != row and entry_is_good(H1[i], H2[i]):    # if H2[i] is inf or nan or 0 then there is a problem in distortion_entry(). SO entry_is_good() is checked
            _avg = distortion_entry(H1[i], H2[i])
            good += 1
            avg  += _avg
    if good > 0:
        avg /= good
    else:
        avg, good = torch.tensor(0., device=device, requires_grad=True), torch.tensor(0., device=device, requires_grad=True)
    return (avg, good)

def distortion(H1, H2, n, jobs=16):
#     dists = Parallel(n_jobs=jobs)(delayed(distortion_row)(H1[i,:],H2[i,:],n,i) for i in range(n))
    dists = (distortion_row(H1[i,:],H2[i,:],n,i) for i in range(n))
    to_stack = [tup[0] for tup in dists]
    avg = torch.stack(to_stack).sum()/n
    return avg
# Does Euclidean to hyperbolic mapping using series of FC layers.
# We use ground truth distance matrix for the pair since the distortion for hyperbolic embs are really low.
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def get_dist_mat(G,node_to_ids):
    lengths = dict(nx.all_pairs_dijkstra_path_length(G))
    # print("lengths = ", lengths)
    n = nx.number_of_nodes(G)
    dist_mat = np.full((n, n), np.inf)

    for i, (node, lengths_to_node) in enumerate(lengths.items()):
        for target_node, length in lengths_to_node.items():
            dist_mat[node_to_ids[node], node_to_ids[target_node]] = length

    return dist_mat


def pairfromidx(data, idx): # data is of class InputData
    # input_tensor = tensorFromSentence(input_vocab, filtered_sentences[idx])
    input_sentence_ids = data.connected_components[idx]
    input_tensor = torch.tensor(list(input_sentence_ids)).view(-1, 1)
    # G = load_graph("random_trees/"+str(idx)+".edges")
    G = nx.from_edgelist(data.trees[idx])
    node_to_ids = {}
    i=0
    for node in input_sentence_ids:
        node_to_ids[node]=i
        i=i+1
    target_matrix = get_dist_mat(G,node_to_ids)
    target_tensor = torch.from_numpy(target_matrix).float().to(device)
    target_tensor.requires_grad = False
    n = G.order()
    # print("input_tensor = ", input_tensor)
    # print("target_tensor = ", target_tensor)
    # print("Order of graph = ", n)
    return (input_tensor.float(), target_tensor, n) # sentences_text[idx] was also returned but idk if its required

    # the word pairfromidx created distance matrix for sentences.
    # for our sentence case -> we will give ids to each seperate tree and run pairfromidx for that tree.
    # well return the

def trainFCHyp(input_matrix, ground_truth, n, mapping, mapping_optimizer, max_length=50):
    mapping_optimizer.zero_grad()

    loss = 0
    # print("input_matrix = ", input_matrix)
    output = mapping(input_matrix.float()) # output shape = input_elements* output_size
    dist_recovered = distance_matrix_hyperbolic(output)
    loss += distortion(ground_truth, dist_recovered, n)
    loss.backward()
    mapping_optimizer.step()

    return loss.item()


# def trainFCIters(data, mapping, n_epochs=5, n_iters=500, print_every=50, plot_every=100, learning_rate=0.01):
#     start = time.time()
#     plot_losses = []
#     print_loss_total = 0
#     plot_loss_total = 0
#     mapping_optimizer = RiemannianSGD(mapping.parameters(), lr=learning_rate, rgrad=poincare_grad, retraction=retraction)
#     # training_pairs = [pairfromidx(idx) for idx in range(n_iters)]
#     training_pairs = [pairfromidx(data, idx) for idx in range(len(data.connected_components))]
#     for i in range(n_epochs):
#         print("Starting epoch "+str(i))
#         iter=1
#         for idx in data.indices:
#             input_matrix = data.euclidean_embeddings[idx]
#             target_matrix = training_pairs[idx][1]
#             n = training_pairs[idx][2]
#             loss = trainFCHyp(input_matrix, target_matrix, n, mapping, mapping_optimizer)
#             print_loss_total += loss
#             plot_loss_total += loss

#             if iter % print_every == 0:
#                 print_loss_avg = print_loss_total / print_every
#                 print_loss_total = 0
#                 print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
#                                              iter, iter / n_iters * 100, print_loss_avg))

#             if iter % plot_every == 0:
#                 plot_loss_avg = plot_loss_total / plot_every
#                 plot_losses.append(plot_loss_avg)
#                 plot_loss_total = 0

#             iter+=1



def trainFCIters2(data, mapping, n_epochs=5, print_every=50, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    mapping_optimizer = RiemannianSGD(mapping.parameters(), lr=learning_rate, rgrad=poincare_grad, retraction=retraction)
    training_pairs = [pairfromidx(data, idx) for idx in range(len(data.connected_components))]
    for epoch in range(n_epochs):
        print("Starting epoch "+str(epoch))
        # Forward pass
        output = mapping(data.eucledian_embeddings_all)
        # loss = mapping_optimizer(output, target)
        # data.euclidean_embeddings has an list of embeddings of sentences in tree
        # training_pairs has the distance matrix for each tree
        # Loss is summed up for all trees
        loss=0
        for i in range(len(training_pairs)):
            dist_recovered = distance_matrix_hyperbolic(output[list(data.connected_components[i])])
            loss += distortion(training_pairs[i][1], dist_recovered, training_pairs[i][2])
        print("loss = ", loss)
        
        # Backward pass
        mapping.zero_grad()
        loss.backward()

        # Update weights
        mapping_optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
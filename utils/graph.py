import numpy as np
import networkx as nx


def get_nodes_in_twohop(graph, train_idx):
    nnodes = len(graph.labels)
    A_tilde = graph.adj_matrix + sp.eye(nnodes)
    idx_vec = np.zeros(nnodes)
    idx_vec[train_idx] += 1
    two_hop = A_tilde @ A_tilde
    twohop_vec = two_hop @ idx_vec
    twohop_idx = np.where(twohop_vec >= 1)[0]
    return twohop_idx


def get_nedges_in_twohop(graph, nx_graph, train_idx):
    nnodes = len(graph.labels)
    idx_vec = np.zeros(nnodes)
    idx_vec[train_idx] += 1
    onehop_vec = idx_vec.T @ graph.adj_matrix
    onehop_idx = np.where(onehop_vec >= 1)[0]
    n_inner_edges = nx.subgraph(nx_graph, onehop_idx).number_of_edges()
    nedges = 0
    for neighbor in onehop_idx:
        nedges += len(list(nx_graph[neighbor]))
    return nedges - n_inner_edges


def construct_line_graph(A):
    """Construct a line graph from an undirected original graph. (from gust)

    Parameters
    ----------
    A : sp.spmatrix [n_samples ,n_samples]
        Symmetric binary adjacency matrix.

    Returns
    -------
    L : sp.spmatrix, shape [A.nnz/2, A.nnz/2]
        Symmetric binary adjancy matrix of the line graph.
    """
    N = A.shape[0]
    edges = np.column_stack(sp.triu(A, 1).nonzero())
    e1, e2 = edges[:, 0], edges[:, 1]

    eye = sp.eye(N).tocsr()
    E1 = eye[e1]
    E2 = eye[e2]

    L = E1.dot(E1.T) + E1.dot(E2.T) + E2.dot(E1.T) + E2.dot(E2.T)

    return L - 2*sp.eye(L.shape[0])


def shortest_kpath(adj_matrix, kwalks):
    eye = sp.eye(adj_matrix.shape[0])
    adjself_matrix = adj_matrix + eye

    T_k = eye
    kpath_mats = []
    for k in range(kwalks):
        T_last = T_k
        T_k = T_k @ adjself_matrix > 0
        kpath_mats.append(T_k - T_last)
    return kpath_mats

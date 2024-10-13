# Find nodes where flow is bigger than threshold (e.g. 5)
# Turn the edge index into adjacency list
# Find neighbors of the suspicious nodes up to hop H
# Turn the edges formed by these nodes into 0
import logging
from collections import defaultdict, deque

import torch


def find_candidate_nodes(edge_index, multiplicity, flow_threshold):
    num_nodes = edge_index.max().item() + 1
    i_zeros = torch.zeros(num_nodes, dtype=torch.float).to(edge_index.device)
    o_zeros = torch.zeros(num_nodes, dtype=torch.float).to(edge_index.device)

    assert edge_index.shape[0] == 2
    incoming = torch.scatter_add(i_zeros, -1, edge_index[1, :], multiplicity.squeeze(-1).to(torch.float))
    outgoing = torch.scatter_add(o_zeros, -1, edge_index[0, :], multiplicity.squeeze(-1).to(torch.float))

    diff = torch.abs(incoming - outgoing)
    nbunch = (diff > flow_threshold).nonzero(as_tuple=True)[0]

    return nbunch.tolist()


def edge_index_to_adj(edge_index):
    adj_list_fw = defaultdict(list)
    adj_list_bw = defaultdict(list)

    for idx, (src, dst) in enumerate(edge_index.t().tolist()):
        adj_list_fw[src].append((dst, idx))
        adj_list_bw[dst].append((src, idx))

    return adj_list_fw, adj_list_bw


def bfs(adj_list, start_node, max_depth):
    visited = set()
    queue = deque([(start_node, 0)])  # (node, depth)

    marked_edges = set()
    while queue:
        node, depth = queue.popleft()
        if node not in visited:
            visited.add(node)
            for neighbor in adj_list[node]:
                node, edge_id = neighbor
                if depth + 1 >= max_depth:
                    continue
                marked_edges.add(edge_id)
                queue.append((node, depth + 1))

    logging.debug(f"Node {start_node} marked {len(marked_edges)}")
    return marked_edges


def unmark_weird_flow(edge_index, multiplicity, flow_threshold, max_depth):
    nbunch = find_candidate_nodes(edge_index, multiplicity, flow_threshold)
    logging.info(f"Considering {len(nbunch)} candidates")
    adj_list_fw, adj_list_bw = edge_index_to_adj(edge_index)
    logging.debug("Converted edge index to adj list")
    marked_edges = set()
    for node in nbunch:
        logging.debug(f"Starting forward search from node {node}")
        marked_edges = marked_edges.union(bfs(adj_list_fw, node, max_depth))
        logging.debug(f"Starting backward search from node {node}")
        marked_edges = marked_edges.union(bfs(adj_list_bw, node, max_depth))

    logging.info(f"Marking {len(marked_edges)} as incorrect")
    marked_edges_idx = torch.tensor(list(marked_edges))
    if marked_edges_idx.numel() > 0:
        multiplicity[marked_edges_idx] = 0.0

    return multiplicity, marked_edges_idx

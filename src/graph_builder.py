from itertools import combinations

import torch
from tqdm import tqdm


class GraphBuilder():
    """TODO (Duncan)

    Attributes:
    """

    def __init__(self, samples):
        """Given the preprocessed training dictionary obtained from preprocessor.py,
        constructs a heuristic graph for each training example.
        """
        self.samples = samples
        self.flat = None  # Flattened current sample

    def build(self):
        """Wrapper to call all graph building functions. Converts edge-lists
        to coorindate pytorch tensors with relation types.

        Returns:
            graphs (torch.Tensor): A 3xN tensor, where N is the sum of the number
                of edges across all graphs in `self.samples`. Here, the first two
                rows correspond to the i-th and j-th indices of an edge
                respectively and the third row corresponds to the relation
                type of the edge (an integer in [0, 1, 2, 3]). The edges and
                corresponding relation types for each graph are concatenated in
                the second dimension to ensure a single tensor can be saved for
                all samples.

            graph_split_sizes (list): A dictionary containing sample ids as keys and
                corresponding graph tensor sizes as values. This allows the
                correct subtensor corresponding to the given sample from `graphs`
                to be extracted during training or inference time by using
                `torch.split()`.
        """
        self._make_names()
        from pprint import pprint
        pprint(self.samples)
        graphs = []
        graph_split_sizes = []

        # Iterate over each training example and build the graph.
        for _, sample in tqdm(self.samples.items()):
            # We only need mention information to build the graph
            sample = sample['mentions']

            # Build graphs.
            doc_based_edges = self._build_doc_based(sample)
            match_edges = self._build_match(sample)
            coref_edges = self._build_coref(sample)
            all_edges = doc_based_edges + match_edges + coref_edges
            comp_edges = self._build_complement(sample, all_edges)

            # Create tensor for each edge specifying relation type.
            rels = torch.LongTensor([0 for _ in range(len(doc_based_edges))] +
                                    [1 for _ in range(len(match_edges))] +
                                    [2 for _ in range(len(coref_edges))] +
                                    [3 for _ in range(len(comp_edges))])

            # Create a coordinate tensor to store edges.
            edge_index = torch.t(torch.LongTensor(doc_based_edges +
                                                  match_edges +
                                                  coref_edges +
                                                  comp_edges))

            graph = torch.cat((edge_index, rels.reshape(1, -1)))
            graphs.append(graph)

            graph_split_sizes.append(graph.shape[1])

        # Concatenate graphs into one large tensor.
        graphs = torch.cat(graphs, dim=-1)

        return graphs, graph_split_sizes

    def _make_names(self):
        """Adds unique indices to each mention for each sample in `self.samples`.
        """

        for sample in self.samples.values():
            idx = 0
            sample_mentions = sample['mentions']

            for doc in sample_mentions:

                for mention in doc:

                    mention['id'] = idx
                    idx += 1

                    for coref in mention['corefs']:
                        coref['id'] = idx
                        idx += 1

    def _make_undirected(self, edge_list):
        """Takes an edge-list and adds to it edges in the reverse direction.
        """
        rev_edge_list = [(edge[1], edge[0]) for edge in edge_list]
        new_edge_list = edge_list + rev_edge_list

        return new_edge_list

    def _build_doc_based(self, sample):
        """Creates an edge-list containing all within-document relationships.
        """
        edge_list = []  # List of edge tuples (id_i, id_j)

        for doc in sample:
            ids = [mention['id'] for mention in doc]
            for mention in doc:
                for coref in mention['corefs']:
                    ids.append(coref['id'])
            edge_list += list(combinations(ids, 2))

        return self._make_undirected(edge_list)

    def _build_match(self, sample):
        """Creates an edge-list containing all mention matches.
        """
        edge_list = []

        checked = set()  # Track IDs that have already had edges added.

        # Get flattened representation of 'sample'.
        flat = []
        for doc in sample:
            for mention in doc:
                flat.append(mention)
                for coref in mention['corefs']:
                    flat.append(coref)
        self.flat = flat

        # Check for case-insensitive matches.
        for mention in flat:
            mention_id = mention['id']
            mention_str = mention['text'].lower()
            if mention_str in checked:
                continue
            checked.add(mention_str)

            for mention2 in flat:
                mention2_id = mention2['id']
                if mention_id == mention2_id:  # Avoid self-loops
                    continue
                mention2_str = mention2['text'].lower()
                if mention_str == mention2_str:  # Positive match
                    edge_list.append((mention_id, mention2_id))

        return self._make_undirected(edge_list)

    def _build_coref(self, sample):
        """Creates an edge-list containing all coref mentions.
        """
        edge_list = []

        for doc in sample:
            for mention in doc:
                ids = [mention['id']]
                for coref in mention['corefs']:
                    ids.append(coref['id'])
                edge_list += list(combinations(ids, 2))

        return self._make_undirected(edge_list)

    def _build_complement(self, sample, all_edges):
        """Creates an edge-list containing complement relations. Here,
        'all_edges' corresponds to an edge list containing all edges
        across all relation types found in the above functions.
        """
        # First get all possible ID pairs.
        all_ids = [mention['id'] for mention in self.flat]
        all_pairs = self._make_undirected(list(combinations(all_ids, 2)))

        # Subtract 'all_edges' from 'all_pairs' to get complement edges.
        comp_edges = list(set(all_pairs) - set(all_edges))

        return comp_edges

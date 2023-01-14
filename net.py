import networkx as nx
from collections import deque
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import random
import uuid
from datetime import datetime


def flow_bfs(G, source):
    fresh_edges = set(G.edges())
    node_queue = deque([source])
    while node_queue:
        node = node_queue.pop()
        for a in G.neighbors(node):
            curr_edge = (node, a)
            if curr_edge in fresh_edges:
                fresh_edges.remove(curr_edge)
                node_queue.append(a)
                yield curr_edge


def n_simple_edges(G, n=1):
    if n <= 0:
        return []
    a = set(list(combinations(range(G.number_of_nodes() - 1), 2)))
    a -= set(G.edges())
    a = {edge if random.random() > 0.5 else (edge[1], edge[0]) for edge in a}
    a = random.sample(a, n)
    return a


class Net:
    def __init__(self, num_inputs, num_outputs, num_nodes=100, num_edges=100):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.master_node = self.num_nodes - 1
        self.input_nodes = list(range(self.master_node - num_inputs, self.master_node))
        self.output_nodes = list(range(num_outputs))
        self.work_nodes = (
            set(list(range(self.num_nodes)))
            - set([self.master_node])
            - set(self.input_nodes)
        )

        self.decay = 0.5
        self.thinking_steps = 10
        self.prune_weight_thresh = 0.01
        self.activated_output = None
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.num_nodes))
        self.G.add_edges_from([(self.master_node, x) for x in self.input_nodes])
        self.G.add_edges_from(n_simple_edges(self.G, n=num_edges))
        self.init_nodes(self.G.nodes())
        self.init_edges(self.G.edges())

    def step(self, inputs, feedback):
        self.backprop(feedback, lr=0.1)

        self.prune()
        self.grow(self.num_edges - self.G.number_of_edges())
        self.mega_prune()

        for node, input in zip(self.input_nodes, inputs):
            self.G.nodes[node]["node_activation"] = input

        for _ in range(self.thinking_steps):
            self.forward()

        act = self.activated_output
        self.activated_outputs = None
        return act

    """
    forward prop: (every brain step, 10x enviro?)
    from used subgraph:
    - run flow_bfs from master node
        for each edge: 
            - if input is activated
            - multiply input by weight, 
            - add that to current activation
            - add current activation to output activation pool
    - check activation
        for each node:
            - if activation input is > 0, set activation to true
            - reset activation pool
    - decay the edges...
    """

    def forward(self):
        # activations = []
        for a, b in flow_bfs(self.G, self.master_node):
            if a != self.master_node:
                self.G.edges[a, b]["edge_activation"] += (
                    self.G.nodes[a]["node_activation"] * self.G.edges[a, b]["weight"]
                )
                self.G.nodes[b]["node_activation"] += self.G.edges[a, b][
                    "edge_activation"
                ]
            # activations.append(self.G.edges[a, b]["edge_activation"])
        # print(f"activation avg is {np.histogram(activations)}")

        for a, b in self.G.edges():
            self.G.edges[a, b]["edge_activation"] *= self.decay

        for node in self.work_nodes:
            if self.G.nodes[node]["node_activation"] >= 1:
                self.G.nodes[node]["node_activation"] = 1
            else:
                self.G.nodes[node]["node_activation"] = 0

        for node in self.output_nodes:
            if self.G.nodes[node]["node_activation"] == 1:
                self.activated_output = node

    """
    if backprop: (checks every env step)
    - reverse digraph
    - from feedback node run flow_bfs
        - for each edge: weight *= tanh(signal * lr * activation_amount) + 1
    """

    def backprop(self, feedback, lr):
        rev_graph = self.G.reverse()
        # for node in [self.activated_output]:
        if self.activated_output is not None:
            node = self.activated_output
            for a, b in flow_bfs(rev_graph, node):
                c, d = b, a  # reverse the reversed edge...
                self.G.edges[c, d]["weight"] *= (
                    np.tanh(feedback * lr * self.G.edges[c, d]["edge_activation"]) + 1
                )

    # prune / grow: (happens when lots of negative rewards?)
    # - trim
    #     - if edge weight < 0.01 delete the edge
    # - create new edges
    #     - create random edges????? might as well try it I guess...
    # - make subgraph of used nodes:
    #     - reverse graph, take bfs tree from each input, add those nodes to used nodes set
    #     - you can use the nodes from this list to a mask
    #         - you can use the mask to wrap the flow-bfs interator, so it would be an iterator with an if statement basically...

    def prune(self):
        low_weight_edges = []
        for edge in self.G.edges():
            if self.G.edges[edge]["weight"] < self.prune_weight_thresh:
                low_weight_edges.append(edge)
        self.G.remove_edges_from(low_weight_edges)
        print(f"pruned {len(low_weight_edges)} edges")

    def grow(self, num_new_edges):
        new_edges = n_simple_edges(self.G, n=num_new_edges)
        new_edges = [(a, b) for a, b in new_edges if b not in self.used_nodes - self.]
        self.G.add_edges_from(new_edges)
        self.init_edges(new_edges)

    # mega-prune:
    # - do backwards bfs's from outputs
    # - this list is all of the used nodes
    # - add the master and input nodes to this list
    # - now you can go through the edges
    #     - if theres an edge that connects to a node not on this list
    #         - delete that edge

    def mega_prune(self):
        rev = self.G.reverse()
        used_nodes = set()
        for out_node in self.output_nodes:
            used_nodes = used_nodes.union(set(nx.bfs_tree(rev, out_node).nodes()))

        unused_nodes = set(self.G.nodes()) - used_nodes
        unused_edges = []
        for a, b in self.G.edges():
            if a in unused_nodes or b in unused_nodes:
                unused_edges.append((a, b))
        self.G.remove_edges_from(unused_edges)

    def init_nodes(self, nodes):
        for node in nodes:
            self.G.nodes[node]["node_activation"] = 0

    def init_edges(self, edges):
        for edge in edges:
            self.G.edges[edge]["weight"] = random.random()
            self.G.edges[edge]["edge_activation"] = 0

    def draw(self):
        G = nx.DiGraph(list(flow_bfs(self.G, source=self.master_node)))
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos)
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.savefig(f"plots/{datetime.now()}.png")
        plt.clf()

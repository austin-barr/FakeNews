import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from utils.data_loader import read_file

class GraphHelper():
    '''Gets features for individual FNN graphs and plots the graphs themselves'''
    
    def __init__(self, dataset):
        '''
        Loads the required graph data for the dataset from data folder
        
        Parameters
        ----------
        dataset : str
            Dataset name, ['politifact', 'gossipcop']
        '''
        self.dataset = dataset
        self.num_graphs = 314 if self.dataset=='politifact' else 5464
        
        folder = f'../data/{self.dataset}/raw/'
        # Edges for each graph as node pairs
        self.edge_index = read_file(folder, 'A', torch.long)
        # Graph number for each node
        self.node_graph_id = np.load(folder + 'node_graph_id.npy').tolist()
        # UNIX timestamp for each retweet node
        timestamps = np.load(f'../data/{self.dataset[0:3]}_id_time_mapping.pkl', allow_pickle=True)
        # Since there aren't timestamps for root nodes, replace the root's time with the first retweet's time
        self.timestamps = np.array([(int(timestamps[i]) if timestamps[i] != '' else int(timestamps[i+1])) for i in range(len(timestamps))])
        
        node_attributes = sp.load_npz(folder + 'new_profile_feature.npz')
        self.profile_features = np.array(node_attributes.todense())
        
    def get_bounds(self, graph_label):
        '''Returns the lower and upper bounds for a graph
        (lower inclusive, upper exclusive)
        
        These are the node numbers that the graph contains, which are used to
        index that graph's data from edge_index, timestamps, and profile_features.
        
        For example, politifact graph 0 has bounds 0 and 497, meaning it contains
        the first 497 nodes of node_graph_id. So timestamps[0:497] will get
        the timestamps for each of the nodes in graph 0
        '''
        lower, upper = self.node_graph_id.index(graph_label), self.node_graph_id.index(graph_label + 1) if graph_label < self.num_graphs-1 else len(self.node_graph_id)
        return lower, upper
    
    def get_edges(self, graph_label):
        '''Returns the edges of a graph as list of node pairs'''
        lower, upper = self.get_bounds(graph_label)
        # Trees have num_nodes - 1 edges so subtract graph_label (1 for each previous graph)
        edges = self.edge_index[lower-graph_label:upper-graph_label-1]
        edges = [[int(i), int(j)] for i, j in edges]
        
        return edges
    
    def get_times_after_root(self, graph_label):
        '''Time differences in seconds between the root and each node in a graph'''
        lower, upper = self.get_bounds(graph_label)
        timestamps = self.timestamps[lower:upper]
        
        first_retweet = timestamps[1]
        timestamps[0] = first_retweet
        
        time_diffs = timestamps - first_retweet
        
        return time_diffs
    
    def get_num_nodes(self, graph_label):
        '''Number of nodes in a graph'''
        lower, upper = self.get_bounds(graph_label)
        
        return upper-lower
    
    def get_lifespan(self, graph_label):
        '''Lifespan of a graph (time between first and last retweet)'''
        lower, upper = self.get_bounds(graph_label)
        
        # upper - 1 since upper is the next graph's root
        lifespan = self.timestamps[upper-1] - self.timestamps[lower]
        
        return lifespan
    
    def get_time_diffs(self, graph_label):
        '''Time differences in seconds between each node and its parent'''
        lower, upper = self.get_bounds(graph_label)
        timestamps = self.timestamps[lower:upper]
        first_retweet = timestamps[1]
        timestamps[0] = first_retweet
        timestamps = np.array(timestamps)
        
        edges = self.get_edges(graph_label)
        labeled_times = {node: time for node, time in zip(range(lower, upper), timestamps)}
        
        time_diffs = []
        for n1, n2 in edges:
            time_diffs.append(labeled_times[n2] - labeled_times[n1])
        
        return time_diffs
    
    def get_avg_time_diff(self, graph_label):
        '''Average time difference in seconds between each node and its parent'''
        time_diffs = self.get_time_diffs(graph_label)
        
        return np.mean(time_diffs)
    
    def get_avg_primary_secondary_time_diff(self, graph_label):
        '''Average time difference in seconds between each primary retweet and its first secondary retweet'''
        lower, upper = self.get_bounds(graph_label)
        timestamps = self.timestamps[lower:upper]
        labeled_times = {node: time for node, time in zip(range(lower, upper), timestamps)}
        edges = self.get_edges(graph_label)
        
        primary = [j for i, j in edges if i == lower]
        secondaries = {p:[] for p in primary}
        
        for n1, n2 in edges:
            if n1 in primary:
                secondaries[n1].append(labeled_times[n2])
        
        primary_diffs = {}
        for node, times in secondaries.items():
            if len(times) != 0:
                primary_diffs[node] = min(times) - labeled_times[node]
        
        avg_time_diff = np.mean([v for v in primary_diffs.values()])
        
        return avg_time_diff
        
    def get_max_depth(self, graph_label):
        '''Maximum depth of the graph's tree'''
        edges = self.get_edges(graph_label)
        G = nx.DiGraph()
        G.add_edges_from(edges)

        depth = nx.dag_longest_path_length(G)
        
        return depth
    
    def get_max_outdegree(self, graph_label):
        '''Maximum node outdegree for a graph'''
        edges = self.get_edges(graph_label)

        G = nx.DiGraph()
        G.add_edges_from(edges)

        max_outdegree = max(G.degree, key=lambda p:p[1])[1]
        
        return max_outdegree
    
    def get_depth_max_outdegree(self, graph_label):
        '''Depth of the node with the highest outdegree in a graph'''
        edges = self.get_edges(graph_label)

        G = nx.DiGraph()
        G.add_edges_from(edges)
        root = edges[0][0]
        
        depth = max(zip(G.degree, nx.shortest_path_length(G, root).items()), key=lambda p: p[0][1])[1][1]
        
        return depth
    
    def get_time_diff_max_outdegree(self, graph_label):
        '''Time difference in seconds between the root and the node with the 
        highest outdegree for a graph'''
        edges = self.get_edges(graph_label)
        time_diffs = self.get_times_after_root(graph_label)

        G = nx.DiGraph()
        G.add_edges_from(edges)
        
        time_diff = max(zip(G.degree, time_diffs), key=lambda p: p[0][1])[1]
        
        return time_diff
        
    def get_profile_feature(self, feature_number, graph_label):
        '''Average of the specified profile feature
        
        Feature names for reference:
        profile_feature_names = ['verified', 'location on', 'follower count',
                    'friend count', 'status count', 'favorites count', 'lists count',
                    'account age', 'name length', 'description length']
        '''
        lower, upper = self.get_node_bounds(graph_label)
        all_features = self.profile_features[lower:upper]
        avg_feature = np.mean([f[feature_number] for f in all_features])
        
        return avg_feature
    
    def draw_graph(self, graph_label, color_times=True, save=True, dpi=400, file_type='png', with_labels=True):
        '''Draws a networkx graph of an FNN graph
        
        Parameters
        ----------
        graph_label : int
            graph label of graph to draw
        color_times : bool, optional
            Whether to use node timestamps to color nodes (default True)
        save : bool, optional
            Whether to save the graph (default is True)
        dpi : int, optional
            Dots per inch of the graph (default is 200)
        file_type : str, optional
            File type for the saved graph
            Any file type supported by matplotlib.plyplot.savefig() (default is 'png')
        with_labels : bool, optional
            Whether to number each node in the graph (default is True)
        '''
        
        G = nx.Graph()
        edges = self.get_edges(graph_label)
        for i, j in edges:
            G.add_edge(i, j)
            
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes)))
        fig, ax = plt.subplots(figsize=(40,40))
        
        if color_times:
            c = self.get_times_after_root(graph_label) / 86400
            cmap = plt.cm.coolwarm
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(c), vmax=max(c)))
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
            cbar.ax.tick_params(labelsize=40)
            cbar.ax.set_xlabel('days', fontsize=40)
            
            nx.draw(G, ax=ax, pos=pos, node_color=c, cmap=cmap, with_labels=True, node_size=[[1000]*len(G.nodes)])
        else:
            nx.draw(G, ax=ax, pos=pos, with_labels=with_labels, node_size=[[1000]*len(G.nodes)])
            
        if save:
            os.makedirs(f'../graphs/{self.dataset}', exist_ok=True)
            plt.savefig(f'../graphs/{self.dataset}/{graph_label}.{file_type}', dpi=dpi)
        
        plt.show()

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from domino import DominoSlicer, explore
import meerkat as mk


# Seeds used by the gnn models, retrieved from here so it's always the same for every model
#[777, 99, 65536, 1864, 1]
seeds = [777]

class RunAnalyzer():
    '''For finding slices, graph feature analysis, making graphs, etc. for a given gnn model run'''
    
    def __init__(self, dataset, model, feature, emb, graph_labels, pred_probs, seed_used):
        '''
        Parameters
        ----------
        dataset : str
            Dataset used, ["gossipcop", "politifact"]
        model : str
            GNN model used, ["upfd", "gcnfn", "gnncl", or "bigcn"]
        feature : str
            Node features used, ["profile", "content", "spacy", "bert"]
        emb : np.ndarray of float, shape (n_graphs, n_features)
            Last layer NN embeddings for each graph
        graph_labels : np.ndarray of int, shape (n_graphs,)
            Labels for each graph. range 0-313 for politifact, 0-5463 for gossipcop
        pred_probs : np.ndarray of float, shape (n_graphs,)
            Softmax class probabilities for each graph (<0.5 is real, >=0.5 is fake)
        seed_used : int
            The random seed value used for the GNN model's run. Used for performance log
        '''
        
        self.dataset = dataset
        self.model = model
        self.feature = feature
        self.emb = emb
        self.graph_labels = graph_labels
        self.pred_probs = pred_probs
        self.pred = np.array([0 if p < 0.5 else 1 for p in self.pred_probs])
        self.seed_used = seed_used
        
        self.info_string = f'{self.dataset} {self.model} {self.feature} seed:{seed_used}'
        
        # Set classes labels based on graph labels and dataset
        if dataset == 'politifact':
            self.class_labels = np.array([0 if i < 157 else 1 for i in self.graph_labels])
        else:
            self.class_labels = np.array([0 if i < 2732 else 1 for i in self.graph_labels])
        
        # Label graphs as 'cr', 'ir', 'cf', 'if' for correct/incorrect real/fake
        self.correctness_class_labels = []
        for graph, c, p in zip(self.graph_labels, self.class_labels, self.pred):
            label = ('c' if c == p else 'i') + ('r' if c == 0 else 'f')
            self.correctness_class_labels.append(label)
    
    def domino_slices(self, random_state, n_slices, n_pca_components=None, save=True):
        '''Gets slices using DominoSlicer

        Parameters
        ----------
        random_state : int
            Random seed to use for initializing the DominoSlicer
        n_slices : int
            Number of slices for DominoSlicer to discover
        n_pca_components : int or None, optional
            Number of PCA components for DominoSlicer to use (default is None)
        save : bool, optional
            Whether to save the slices as an npy file (default is True)
        
        Returns
        -------
        slices : np.ndarray of int, shape (n_graphs, n_slices)
            1hot encoded matrix from DominoSlicer

        '''
        # Create directory for saving
        if save:
            os.makedirs('slices', exist_ok=True)
        
        # Meerkat data panel to use for slicing
        dp = mk.DataPanel (
            {
                'embedding': self.emb,
                'target': self.class_labels,
                'pred_probs': self.pred_probs,
            }
        )
        # Get slices
        slicer = DominoSlicer(n_pca_components=n_pca_components, random_state=random_state, n_slices=n_slices)
        slicer.fit(data=dp)
        slices = slicer.predict(data=dp)
        
        if save:
            np.save('slices/slices.npy', slices)
        
        # Log slice performance
        slice_labels = np.where(slices==1)[1]
        unique_slices = set(slice_labels)
        with open('slice_performance_log.txt', 'a') as outfile:
            outfile.write(f'{self.info_string}\n')
            outfile.write(f'random_state:{random_state} n_slices:{n_slices}\n')
            
            base_acc = metrics.accuracy_score(self.class_labels, self.pred)
            base_f1 = metrics.f1_score(self.class_labels, self.pred)
            print(f'base performance: {base_acc:.4f}, {base_f1:.4f}')
            outfile.write(f'base performance: {base_acc:.4f}, {base_f1:.4f}\n\n')
            for s in unique_slices:
                slice_class_labels = self.class_labels[slice_labels==s]
                slice_pred = self.pred[slice_labels==s]
                
                slice_acc = metrics.accuracy_score(slice_class_labels, slice_pred)
                slice_f1 = metrics.f1_score(slice_class_labels, slice_pred)
                
                num_in_slice = len(slice_class_labels)
                cr = sum([1 for i in range(num_in_slice) if slice_class_labels[i] == 0 and slice_pred[i] == 0])
                ir = sum([1 for i in range(num_in_slice) if slice_class_labels[i] == 0 and slice_pred[i] == 1])
                cf = sum([1 for i in range(num_in_slice) if slice_class_labels[i] == 1 and slice_pred[i] == 1])
                icf = num_in_slice - cr - ir - cf
                
                cr /= num_in_slice / 100
                ir /= num_in_slice / 100
                cf /= num_in_slice / 100
                icf /= num_in_slice / 100
                
                print(f'{s} {slice_acc:.4f} {slice_f1:.4f}')
                print(f' {num_in_slice} cr:{cr:.2f}% ir:{ir:.2f}% cf:{cf:.2f}% if:{icf:.2f}%')
                outfile.write(f'{s} {slice_acc:.4f} {slice_f1:.4f}\n')
                outfile.write(f' {num_in_slice} cr:{cr:.2f}% ir:{ir:.2f}% cf:{cf:.2f}% if:{icf:.2f}%\n\n')
            
        return slices
    
    def domino_explore(self, slices):
        '''Uses the domino explore feature on discovered slices
        
        Explore is built to run in original Jupyter Notebook, so this should be
        called from a file running in that
        
        Parameters
        ----------
        slices : np.ndarray of int, shape (n_graphs, n_slices)
            1hot encoded matrix from DominoSlicer
        '''
        dp = mk.DataPanel(
            {
                'embedding': self.emb,
                'target': self.class_labels,
                'pred_prob': self.pred_probs,
                'slices': slices
            }
        )
        
        explore(data=dp)
    
    def graph_slices(self, slices, dim_reduction='pca', save=True, dpi=1200, file_type='png'):
        '''Makes a 2D graph of embeddings showing the graphs in each slice
        
        Parameters
        ----------
        slices : np.ndarray of int, shape (n_graphs, n_slices)
            1hot encoded matrix from DominoSlicer
        dim_reduction : str, optional
            Type of dimensionality reduction, ['pca', 'tsne'] (default is 'pca')
        save : bool, optional
            Whether to save the graph (default is True)
        dpi : int, optional
            Dots per inch of the graph (default is 1200)
        file_type : str, optional
            File type for the saved graph
            Any file type supported by matplotlib.plyplot.savefig() (default is 'png')
        '''
        # Create directory for saving
        if save:
            os.makedirs('slices', exist_ok=True)
        
        slice_labels = np.where(slices==1)[1]
        
        colors = ['b', 'g', 'r', 'c', 'y', 'k', 'orange', 'violet', 'grey', 'lime', 'bisque']
        
        c = [colors[label] for label in slice_labels]
        
        used_colors = set(c)
        
        if dim_reduction == 'pca':
            pca = PCA()
            emb = pca.fit_transform(self.emb)[:,:2]
        elif dim_reduction == 'tsne':
            tsne = TSNE(perplexity=10, n_iter=360)
            emb = tsne.fit_transform(emb)
        plt.scatter(emb[:,0], emb[:,1], c=c)
        plt.title('slices')
        
        color_labels = [f'slice {colors.index(color)} ({sum(slice_labels==i)})' for i, color in enumerate(colors) if color in used_colors]
        color_handles = [plt.scatter([], [], color=color) for color in colors if color in used_colors]
        plt.legend(color_handles, color_labels)
        
        if save:
            plt.savefig(f'slices/slices_graph.{file_type}', dpi=dpi)
        
        plt.show()
        
    def get_feature_dict(self, feature_name, feature_function, slices, save=True):
        '''Creates a dictionary for a graph feature

        Parameters
        ----------
        feature_name : str
            Name of the feature being retrieved, used in the save file name
        feature_function : function(graph_label) -> graph_feature
            The function to be called for each graph label
        slices : np.ndarray of int, shape (n_graphs, n_slices)
            1hot encoded matrix from DominoSlicer
        save : bool, optional
            Whether to save the feature dict as an npy, (default is True)

        Returns
        -------
        feature_dict : dict of {str: list}
            'cr', 'ir', 'cf', 'if' are used as indices for correct/incorrect real/fake,
            which contains the features of each graph in that group separated by slice
            These are of shape (n_slices, n_graphs_in_slice)
            
            The 'all' index has the feature for all graphs while preserving order
            'all' list is of shape (n_graphs,)
        '''
        # Create directory for saving
        if save:
            os.makedirs('saved_features', exist_ok=True)
        
        # Convert from 1hot to 1D array
        slice_labels = np.where(slices==1)[1]
        
        num_slices = slices.shape[1]
        feature_dict = {label: [[] for i in range(num_slices)] for label in ('cr','ir','cf','if')}
        feature_dict['all'] = []
        
        # Get features for each graph and store them in feature_dict
        for graph, label, s in zip(self.graph_labels, self.correctness_class_labels, slice_labels):
            feature = feature_function(graph)
            feature_dict[label][s].append(feature)
            feature_dict['all'].append(feature)
            
        if save:
            np.save(f'saved_features/{feature_name.replace(" ", "_")}.npy', feature_dict)
        
        return feature_dict
    
    def slice_analysis(self, feature_name, feature_dict, slices, is_profile, y_label='', showfliers=False, showmeans=False, save=True, file_type='png'):
        '''Makes boxplots of a feature
        
        Parameters
        ----------
        feature_name : str
            Name of feature used
        feature_dict : dict of {str: list}
            Feature dictionary, see get_feature_dict()
        slices : np.ndarray of int, shape (n_graphs, n_slices)
            1hot encoded matrix from DominoSlicer
        is_profile : bool
            Whether the feature is a profile feature, which are saved to profile folder
        y_label : str, optional
            y label for feature (default '')
        showfliers : bool, optional
            Value for showfliers on plots (default False)
        showmeans : bool, optional
            Value for showmeans on plots (default False)
        save : bool, optional
            Whether to save the graphs (default True)
        file_type : str, optional
            File type of saved graphs (default is png)
        '''
        
        title = feature_name
        # String for file names
        feature = feature_name.replace(' ', '_')
        
        if is_profile:
            root = f'slices/profile/{feature}'
        else:
            root = f'slices/{feature}'
            
        # Create directory for saving
        if save:
            os.makedirs(root, exist_ok=True)
        
        # convert from 1hot to 1D array
        slice_labels = np.where(slices==1)[1]
        
        
        
        unique_slices = sorted(set(slice_labels))
        
        # real vs fake
        real = flatten(feature_dict['cr']) + flatten(feature_dict['ir'])
        fake = flatten(feature_dict['cf']) + flatten(feature_dict['if'])
        
        plt.boxplot([real, fake], labels=['real', 'fake'], showfliers=showfliers, showmeans=showmeans)
        
        plt.title(f'{title}')
        plt.ylabel(y_label)
        
        if save:
            plt.savefig(f'{root}/{feature}.{file_type}')
        
        plt.show()
        
        # feature by slice
        for s in unique_slices:
            features = np.array(feature_dict['all'])[slice_labels==s]
            plt.boxplot(features, positions=[s], labels=[f'{s} ({len(features)})'], showfliers=showfliers, showmeans=showmeans)
        
        plt.title(f'{title} by slice')
        plt.ylabel(y_label)
        
        if save:
            plt.savefig(f'{root}/{feature}_by_slice.{file_type}')
        
        plt.show()
        
        # real features for each slice separated by correct/incorrect
        for s, features in zip(unique_slices, feature_dict['cr']):
            plt.boxplot(features, positions=[s*2], labels=[f'{s} ({len(features)})'], showfliers=showfliers, showmeans=showmeans)
        
        for s, features in zip(unique_slices, feature_dict['ir']):
            plt.boxplot(features, positions=[s*2+1], labels=[f'({len(features)})'], showfliers=showfliers, showmeans=showmeans)
        
        plt.title(f'real {title} by slice with correct/incorrect')
        plt.ylabel(y_label)
        
        if save:
            plt.savefig(f'{root}/real_{feature}_by_slice_with_correct.{file_type}')
        
        plt.show()
        
        # real features for each slice
        for s, cr, ir in zip(unique_slices, feature_dict['cr'], feature_dict['ir']):
            plt.boxplot(cr+ir, positions=[s], labels=[f'{s} ({len(cr+ir)})'], showfliers=showfliers, showmeans=showmeans)
            
        plt.title(f'real {title} by slice')
        plt.ylabel(y_label)
        
        if save:
            plt.savefig(f'{root}/real_{feature}_by_slice.{file_type}')
        
        plt.show()
        
        for s, features in zip(unique_slices, feature_dict['cf']):
            plt.boxplot(features, positions=[s*2], labels=[f'{s} ({len(features)})'], showfliers=showfliers, showmeans=showmeans)
        
        for s, features in zip(unique_slices, feature_dict['if']):
            plt.boxplot(features, positions=[s*2+1], labels=[f'({len(features)})'], showfliers=showfliers, showmeans=showmeans)
        
        plt.title(f'fake {title} by slice with correct/incorrect')
        plt.ylabel(y_label)
        
        if save:
            plt.savefig(f'{root}/fake_{feature}_by_slice_with_correct.{file_type}')
        
        plt.show()
        
        # fake features for each slice
        for s, cf, icf in zip(unique_slices, feature_dict['cf'], feature_dict['if']):
            plt.boxplot(cf+icf, positions=[s], labels=[f'{s} ({len(cf+icf)})'], showfliers=showfliers, showmeans=showmeans)
        
        plt.title(f'fake {title} by slice')
        plt.ylabel(y_label)
        
        if save:
            plt.savefig(f'{root}/fake_{feature}_by_slice.{file_type}')
        
        plt.show()
        
        # real features correct/incorrect
        for i, features in enumerate([flatten(feature_dict['cr']), flatten(feature_dict['ir'])]):
            plt.boxplot(features, positions=[i], labels=[f'{["correct", "incorrect"][i]} ({len(features)})'], showfliers=showfliers, showmeans=showmeans)
        plt.title(f'real {title} with correct/incorrect')
        plt.ylabel(y_label)
        
        if save:
            plt.savefig(f'{root}/real_{feature}_with_correct.{file_type}')
        
        plt.show()
        
        # fake features correct/incorrect
        for i, features in enumerate([flatten(feature_dict['cf']), flatten(feature_dict['if'])]):
            plt.boxplot(features, positions=[i], labels=[f'{["correct", "incorrect"][i]} ({len(features)})'], showfliers=showfliers, showmeans=showmeans)
        
        plt.title(f'fake {title} with correct/incorrect')
        plt.ylabel(y_label)
        
        if save:
            plt.savefig(f'{root}/fake_{feature}_with_correct.{file_type}')
        
        plt.show()
        
        # features by prediction confidence
        colors = ['b', 'g', 'r', 'c', 'y', 'k', 'orange', 'violet', 'grey', 'lime', 'bisque']
        sorted_all = sorted(zip(feature_dict['all'], slice_labels, self.class_labels, self.pred_probs), key=lambda x: x[-1])
        real_features, real_slice_labels, real_prob = [], [], []
        fake_features, fake_slice_labels, fake_prob = [], [], []
        for feat, slice_label, class_label, prob in sorted_all:
            if class_label == 0:
                real_features.append(feat)
                real_slice_labels.append(slice_label)
                real_prob.append(prob)
            else:
                fake_features.append(feat)
                fake_slice_labels.append(slice_label)
                fake_prob.append(prob)
        
        # real features by prediction confidence
        c = []
        for i, label in enumerate(real_slice_labels):
            c.append(colors[label])
        plt.scatter(real_prob, real_features, c=c, alpha=0.5)
        plt.title(f'real {title}')
        plt.xlabel('prediction')
        plt.ylabel(y_label)
        
        unique_slices = sorted(set(slice_labels))
        
        color_labels = [f'slice {s}' for s in unique_slices]
        color_handles = [plt.scatter([], [], color=color) for color in colors]
        plt.legend(color_handles, color_labels)
        
        plt.savefig(f'{root}/real_{feature}_by_confidence.{file_type}')
        
        plt.show()
        
        # fake features by prediction confidence
        c = []
        for label in fake_slice_labels:
            c.append(colors[label])
        plt.scatter(fake_prob, fake_features, c=c, alpha=0.5)
        plt.title(f'fake {title}')
        plt.xlabel('confidence')
        plt.ylabel(y_label)
        
        color_labels = [f'slice {s}' for s in unique_slices]
        color_handles = [plt.scatter([], [], color=color) for color in colors]
        plt.legend(color_handles, color_labels)
        
        if save:
            plt.savefig(f'{root}/fake_{feature}_by_confidence.{file_type}')
        
        plt.show()
        
    def poster_graphs(self, slices, lifespan_dict):
        '''Makes graphs for posters

        Parameters
        ----------
        slices : np.ndarray of int, shape (n_graphs, n_slices)
            1hot encoded matrix from DominoSlicer
        lifespan_dict : dict of {str: list}
            Feature dict of graph lifespans, see get_feature_dict()

        '''
        # convert from 1hot to 1D array
        slice_labels = np.where(slices==1)[1]
        unique_slices = set(slice_labels)
    
        # PCA with slice 0
        pca = PCA()
        emb = pca.fit_transform(self.emb)[:,:2]
        
        colors = ['b', 'g', 'r', 'c', 'y', 'k', 'orange', 'violet', 'grey', 'lime', 'bisque']
        c = [colors[label] for label in self.slice_labels]
        used_colors = set(c)
        
        plt.scatter(self.pred_probs, emb[:,1], c=c, alpha=0.5)
        plt.title('Domino Slices with PCA of GNN Embeddings')
        
        accs = []
        for s in unique_slices:
            slice_class_labels = self.class_labels[slice_labels==s]
            slice_pred = self.pred[slice_labels==s]
            slice_acc = metrics.accuracy_score(slice_class_labels, slice_pred) * 100
            accs.append(slice_acc)
            
        color_labels = [f'Slice {colors.index(color)} ({accs[i]:.2f}%)' for i, color in enumerate(colors) if color in used_colors]
        color_handles = [plt.scatter([], [], color=color) for color in colors if color in used_colors]
        plt.legend(color_handles, color_labels)
        
        plt.xlabel('Model Prediction')
        plt.xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=['0.0\n(Real)', '0.2', '0.4', '0.6', '0.8', '1.0\n(Fake)'])
        
        plt.savefig('slices/poster/pca_with_slices.png', bbox_inches='tight', dpi=2400)
        
        plt.show()
        
        # lifespans class difference with slice 0
        all_real = flatten(lifespan_dict['cr'] + lifespan_dict['ir'])
        all_fake = flatten(lifespan_dict['cf'] + lifespan_dict['if'])
        s0_cr = lifespan_dict['cr'][0]
        s0_ir = lifespan_dict['ir'][0]
        s0_cf = lifespan_dict['cf'][0]
        s0_if = lifespan_dict['if'][0]
        
        fig, ax = plt.subplots()

        # real boxes
        ax.boxplot([all_real, s0_cr, s0_ir], positions=[1, 3, 4], widths=0.6, showfliers=False, showmeans=True)

        # fake boxes
        ax.boxplot([all_fake, s0_cf, s0_if], positions=[2, 5, 6], widths=0.6, showfliers=False, showmeans=True)

        # lines between groups
        ax.axvline(x=2.5, color='k', linestyle='-', linewidth=1)
        ax.axvline(x=4.5, color='k', linestyle=':', linewidth=1)
        
        # labels
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.set_xticklabels(['Real', 'Fake', 'Correct', 'Incorrect', 'Correct', 'Incorrect'])
        ax.set_ylabel('Values')
        ax.set_title('Box Plot with Grouped Data')
        
        plt.annotate('All Graphs', (1/6,-0.2), xycoords='axes fraction', ha='center')
        plt.annotate('Slice 0 Graphs', (4/6,-0.2), xycoords='axes fraction', ha='center')
        
        plt.annotate('Real', (3/6,-0.15), xycoords='axes fraction', ha='center')
        plt.annotate('Fake', (5/6,-0.15), xycoords='axes fraction', ha='center')
        
        plt.title('News Lifespan on Twitter')
        plt.ylabel('days')
        
        plt.savefig('slices/poster/lifespan.png', bbox_inches='tight', dpi=2400)
        plt.show()
    
def get_exclude(dataset, emb, pred, graph_labels, num_exclude, dim_reduction=None, show_graph=True, graph_dim_reduction='pca', title='', save_graph=True, dpi=1200, file_type='png'):
    '''Returns the labels of graphs to exclude from the training set
    
    Also makes a 2D plot of graph embeddings showing class, correctness, and excluded points
    
    Parameters
    ----------
    dataset : str
        The dataset being used, ["gossipcop", "politifact"]
        Needed to generate class_labels
    emb : np.ndarray of float, shape (n_graphs, n_features)
        Last layer NN embeddings for each graph
    pred : np.ndarray of int, shape (n_graphs,)
        Model prediction for each graph. 0 for real, 1 for fake
    graph_labels : np.ndarray of int, shape (n_graphs,)
        Labels for each graph. range 0-313 for politifact, 0-5463 for gossipcop
    num_exclude : int
        Max number of graphs to exclude
    dim_reduction : str or None, optional
        Type of dimensionality reduction. Either 'pca', 'tsne', or None for no reduction
        (default is None)
    show_graph : bool, optional
        Whether to plot pca/tsne graph showing classes, predictions, and excluded graphs
        (default is True)
    graph_dim_reduction : str, optional
        Type of dimensionality reduction to use for the graph. Either 'pca' or 'tsne'
        (default is 'pca')
    title : str
        Title of the 2D graph (default is '')
    save_graph : bool, optional
        Whether to save the created graph (default is True)
    dpi : int, optional
        Dots per inch of the graph (default is 1200)
    file_type : str, optional
        File type for the saved graph
        Any file type supported by matplotlib.plyplot.savefig() (default is 'png')
        
    Returns
    -------
    exclude : list
        Graph labels to exclude selected from graph_labels, ordered furthest from the 
        opposite class's center to closest
    '''
    # Create directory for saving
    if save_graph:
        os.makedirs('last_run', exist_ok=True)
    
    if dim_reduction == 'tsne':
        tsne = TSNE(perplexity=10, n_iter=360)
        emb = tsne.fit_transform(emb)
    elif dim_reduction == 'pca':
        pca = PCA()
        emb = pca.fit_transform(emb)[:,:2]
    
    # DBSCAN
    # an attempt to use KNN and DBSCAN to find clusters on 2D graphs (not for full embeddings)
    # gets eps value for DBSCAN automatically using KNN and KneeLocator
    # doesn't work too well, but it's here for reference if you want to use DBSCAN ig
    # a good eps value can be found for a specific run by just testing values
    '''
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k).fit(emb)
    distances, _ = nbrs.kneighbors(emb)
    k_distances = distances[:, -1]
    
    k_distances_sorted = np.sort(k_distances)
    
    knee = KneeLocator(range(len(k_distances)), k_distances_sorted, curve='convex', direction='increasing')
    optimal_eps = knee.elbow_y
    
    print(f"Optimal eps: {optimal_eps}")
    
    knee.plot_knee()
    plt.show()
    
    dbscan = DBSCAN(eps=optimal_eps)
    cluster_labels = dbscan.fit_predict(emb)
    
    print(f'{len(np.unique(cluster_labels))} clusters')
    
    c = [f'C{label}' if label != -1 else 'black' for label in cluster_labels]
    
    plt.scatter(emb[:,0], emb[:,1], c=c)
    plt.show()
    '''
    if dataset == 'politifact':
        num_graphs = 314
    elif dataset == 'gossipcop':
        num_graphs = 5464
    class_labels = [0 if i < num_graphs/2 else 1 for i in graph_labels]
    class_labels = np.array(class_labels)
    
    real_emb = emb[class_labels == 0]
    fake_emb = emb[class_labels == 1]
    
    real_center = np.mean(real_emb, axis=0)
    fake_center = np.mean(fake_emb, axis=0)
    emb_centers = np.vstack((real_center, fake_center))
    
    distances = pairwise_distances(emb, emb_centers)
    
    incorrect_indices = [i for i in range(len(class_labels)) if pred[i] != class_labels[i]]
    
    # sort graph labels by distance from the opposite class's center, selecting the furthest ones based on num_exclude
    exclude = graph_labels[sorted(incorrect_indices, key=lambda i: distances[i][class_labels[i]], reverse=True)][:num_exclude]
    
    if show_graph or save_graph:
        if dim_reduction != None:
            graph_emb = emb
        elif graph_dim_reduction == 'tsne':
            tsne = TSNE(perplexity=10, n_iter=360)
            graph_emb = tsne.fit_transform(emb)
        elif graph_dim_reduction == 'pca':
            pca = PCA()
            graph_emb = pca.fit_transform(emb)[:,:2]
        
        point_colors = ['C0', 'C2', 'C1', 'C3']
        c = []
        for i in range(len(class_labels)):
            if class_labels[i] == 0:
                if pred[i] == 0:
                    c.append(point_colors[0])
                else:
                    c.append(point_colors[1])
            else:
                if pred[i] == 1:
                    c.append(point_colors[2])
                else:
                    c.append(point_colors[3])
                    
        s = [100 if label in exclude else 20 for label in graph_labels]
        
        plt.scatter(graph_emb[:,0], graph_emb[:,1], c=c, s=s)
        
        reduced_real_points = [emb for i, emb in enumerate(graph_emb) if class_labels[i] == 0 and pred[i] == 0]
        reduced_fake_points = [emb for i, emb in enumerate(graph_emb) if class_labels[i] == 1 and pred[i] == 1]
        
        reduced_real_center = np.mean(reduced_real_points, axis=0)
        reduced_fake_center = np.mean(reduced_fake_points, axis=0)
        
        if len(reduced_real_points) != 0:
            plt.scatter(reduced_real_center[0], reduced_real_center[1], marker='*', color='black')
        if len(reduced_fake_points) != 0:
            plt.scatter(reduced_fake_center[0], reduced_fake_center[1], marker='*', color='black')
        
        plt.title(title)
        
        num_correct_real = sum([1 for i in range(len(graph_emb)) if class_labels[i] == 0 and pred[i] == 0])
        num_incorrect_real = sum([1 for i in range(len(graph_emb)) if class_labels[i] == 0 and pred[i] == 1])
        num_correct_fake = sum([1 for i in range(len(graph_emb)) if class_labels[i] == 1 and pred[i] == 1])
        num_incorrect_fake = sum([1 for i in range(len(graph_emb)) if class_labels[i] == 1 and pred[i] == 0])
        
        color_labels = [f'correct real ({num_correct_real})',
                        f'incorrect real ({num_incorrect_real})',
                        f'correct fake ({num_correct_fake})',
                        f'incorrect fake ({num_incorrect_fake})']
        
        color_handles = [plt.scatter([], [], color=color) for color in point_colors]
        plt.legend(color_handles, color_labels)
        
        if save_graph:
            plt.savefig(f'exclude_graph.{file_type}', dpi=dpi)
        if show_graph:
            plt.show()
    
    return exclude

def flatten(arr):
    return [item for sublist in arr for item in sublist]

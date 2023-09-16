import numpy as np

from utils.graph_helper import GraphHelper
from utils.run_helper import RunAnalyzer

'''File to run for slicing and feature analysis after running a gnn model

- Loads saved data from the last gnn run
- Gets domino slices
- Saves and loads feature dictionaries for slice feature graphs

New features can be added in the same way as others,
just call get_feature_dict with the new feature function and name

Just comment stuff out depending on what you need to do
'''

# Load last run's saved data
root = 'last_run'
settings = np.load(f'{root}/settings.npy', allow_pickle=True).item()
dataset = settings['dataset']
model = settings['model']
feature = settings['feature']
seed_used = settings['seed']

pred_probs = np.load(f'{root}/pred_probs.npy')
emb = np.load(f'{root}/emb.npy')
graph_labels = np.load(f'{root}/graph_labels.npy')

# RunAnalyzer object to generate features for last run
ra = RunAnalyzer(dataset, model, feature, emb, graph_labels, pred_probs, seed_used)

'''Generate slices if needed'''
random_state = 4
n_slices = 5
slices = ra.domino_slices(random_state, n_slices, save=True)

'''otherwise just load saved slices'''
slices = np.load('slices/slices.npy')

# GraphHelper object for feature retrieval functions
gh = GraphHelper(dataset)

# lifespans
def lifespan():
    def feature_function(graph_label):
        return gh.get_lifespan(graph_label) / 86400 # convert to days
    feature = 'lifespan'
    ra.get_feature_dict(feature, feature_function, slices)

# profile features
def profile(feature_numbers):
    for feature_number in feature_numbers:
        def feature_function(graph_label):
            return gh.get_profile_feature(feature_number, graph_label)
        
        feature = feature_names[feature_number]
        ra.get_feature_dict(feature, feature_function, slices)
        
# number of nodes
def num_nodes():
    feature_function = gh.get_num_nodes
    feature = 'num nodes'
    ra.get_feature_dict(feature, feature_function, slices)

# tree depth (S1)
def depth():
    feature_function = gh.get_max_depth
    feature = 'graph depth'
    ra.get_feature_dict(feature, feature_function, slices)

# max outdegree (S3)
def max_outdegree():
    feature_function = gh.get_max_outdegree
    feature = 'max outdegree'
    ra.get_feature_dict(feature, feature_function, slices)

# depth of max outdegree (S5)
def depth_of_max_outdegree():
    feature_function = gh.get_depth_max_outdegree
    feature = 'depth of max outdegree'
    ra.get_feature_dict(feature, feature_function, slices)

# time until node with max outdegree (T3)
def time_until_max_outdegree():
    def feature_function(graph_label):
        return gh.get_time_diff_max_outdegree(graph_label) / 86400
    
    feature = 'time until max outdegree'
    ra.get_feature_dict(feature, feature_function, slices)

# avg time diff between all retweets (~T7)
def avg_time_diff():
    def feature_function(graph_label):
        return gh.get_avg_time_diff(graph_label) / 86400
    
    feature = 'avg time difference'
    ra.get_feature_dict(feature, feature_function, slices)

# avg time diff between each primary and first secondary retweet (~T8)
def avg_primary_secondary_time_diff():
    def feature_function(graph_label):
        return gh.get_avg_primary_secondary_time_diff(graph_label) / 86400
    
    feature = 'avg primary secondary time difference'
    ra.get_feature_dict(feature, feature_function, slices)

'''Save feature dicts to npy files through get_feature_dict() since they take a while to get
Comment out after they're saved'''
# profile(range(10))
lifespan()
# num_nodes()
# depth()
# max_outdegree()
# depth_of_max_outdegree()
# time_until_max_outdegree()
# avg_time_diff()
# avg_primary_secondary_time_diff()

# Graph slices
ra.graph_slices(slices)

'''
profile_feature_names = ['verified', 'geo-spatial positioning enabled', 'follower count',
            'friend count', 'status count', 'favorites count', 'number of lists',
            'account age', 'words in name', 'description length']
'''
# feature names for reference
profile_feature_names = ['verified', 'location on', 'follower count',
            'friend count', 'status count', 'favorites count', 'lists count',
            'account age', 'name length', 'description length']

feature_names = ['lifespan', 'num nodes', 'max outdegree', 'depth of max outdegree',
                 'time until max outdegree', 'avg time difference', 'avg primary secondary time difference']

# Load feature dict from saved npy
feature = 'lifespan'
feature_dict = np.load(f'saved_features/{feature.replace(" ", "_")}.npy', allow_pickle=True).item()

if feature in profile_feature_names:
    is_profile = True
else:
    is_profile = False

y_label = ''
ra.slice_analysis(feature, feature_dict, slices, is_profile=is_profile, y_label=y_label)

'''Domino explore feature, meant to run in Jupyter notebook'''
ra.domino_explore(slices)

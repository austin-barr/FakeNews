import argparse
import copy as cp
import os
import sys
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from tqdm import tqdm

from utils.data_loader import *
from utils.eval_helper import *

import utils.run_helper as run_helper

"""

Modified version of BiGCN implementation from:

Paper: User Preference-aware Fake News Detection
Link: https://arxiv.org/pdf/2104.12259.pdf
Source code: https://github.com/safe-graph/GNN-FakeNews/

    The Bi-GCN is adopted from the original implementation from the paper authors 
    
    Paper: Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks
    Link: https://arxiv.org/pdf/2001.06362.pdf
    Source Code: https://github.com/TianBian95/BiGCN

"""

class TDrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = cp.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = cp.copy(x)
        rootindex = data.root_index
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(rootindex.device)
        batch_size = max(data.batch) + 1

        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(rootindex.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)

        return x

class BUrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = cp.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = cp.copy(x)

        rootindex = data.root_index
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(rootindex.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(rootindex.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = scatter_mean(x, data.batch, dim=0)
        return x

class Net(torch.nn.Module):
    # For tracking last layer embeddings and predictions for each call to forward()
    # base was used for data removal to indicate a run with the unmodified training set
    embeddings = []
    pred_probs = []
    base = True
    def __init__(self, in_feats, hid_feats, out_feats):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.fc = torch.nn.Linear((out_feats+hid_feats) * 2, 2)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = torch.cat((TD_x, BU_x), 1)
        x = self.fc(x)
        
        # use embeddings from last layer before log softmax
        emb = x.cpu().detach().numpy()
        Net.embeddings.append(emb)
        
        x = F.log_softmax(x, dim=-1)
        
        # reverse log with exp to get softmax class probabilities
        pred_probs = torch.exp(x).cpu().detach().numpy()[:,1]
        Net.pred_probs.append(pred_probs)
        
        return x

def compute_test(loader, verbose=False):
    model.eval()
    loss_test = 0.0
    out_log = []
    with torch.no_grad():
        for data in loader:
            if not args.multi_gpu:
                data = data.to(args.device)
            out = model(data)
            if args.multi_gpu:
                y = torch.cat([d.y for d in data]).to(out.device)
            else:
                y = data.y
            if verbose:
                print(F.softmax(out, dim=1).cpu().numpy())
            out_log.append([F.softmax(out, dim=1), y])
            loss_test += F.nll_loss(out, y).item()
    return eval_deep(out_log, loader), loss_test

'''
datasets = ["politifact", "gossipcop"]
features = ["profile", "content", "spacy", "bert"]
'''
# which dataset/feature(s) to run
datasets = ["gossipcop"]
features = ["content"]

# How many groups exclude is split into for gradual removal
# 1 excludes all at once, 2 excludes half then all, 3 excludes 1/3 then 2/3 then all, etc
# -1 sets this to len(exclude), so each graph is excluded individually
# 0 turns off removal
num_splits = 2

# Max number of graphs to exclude
num_exclude = 20

# Saves test embeddings, pred probs, and graph labels for slicing
save_for_slices = True

# Toggles early stopping
early_stopping = True

# Learning rate and number of epochs I found worked best (with early stopping)
parameters = {'politifact profile': (0.001, 50),
              'politifact content': (0.001, 150),
              'politifact spacy': (0.01, 50),
              'politifact bert': (0.01, 50),
              'gossipcop profile': (0.001, 50),
              'gossipcop content': (0.001, 50),
              'gossipcop spacy': (0.001, 50),
              'gossipcop bert': (0.001, 50),
              }

for ds in datasets:
    for ftr in features:
        seeds = run_helper.seeds
        lr, epochs = parameters[f'{ds} {ftr}']
        
        # for tracking accuracy, f1 and number of graphs excluded
        # full is everything in exclude removed, best is based on highest f1 for each seed
        base_performances = []
        full_excl_performances = []
        best_excl_performances = []
        full_excl_num = []
        best_excl_num = []
        
        for seed in seeds:
            # accuracy and f1 for each exclude split with this seed
            seed_excl_performances = []
            exclude = []
            Net.base = True
            # counts the number of splits of exclude that have been used
            step = 0
            while step <= num_splits or Net.base:
                
                val_losses = [0] * epochs
                train_losses = [0] * epochs
                val_accuracies = [0] * epochs
                models = [0] * epochs
                
                parser = argparse.ArgumentParser()
                
                parser.add_argument('--seed', type=int, default=seed, help='random seed')
                parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
                # hyper-parameters
                parser.add_argument('--dataset', type=str, default=ds, help='[politifact, gossipcop]')
                parser.add_argument('--batch_size', type=int, default=128, help='batch size')
                parser.add_argument('--lr', type=float, default=lr, help='learning rate')
                parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
                parser.add_argument('--nhid', type=int, default=128, help='hidden size')
                parser.add_argument('--TDdroprate', type=float, default=0.2, help='dropout ratio')
                parser.add_argument('--BUdroprate', type=float, default=0.2, help='dropout ratio')
                parser.add_argument('--epochs', type=int, default=epochs, help='maximum number of epochs')
                parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
                parser.add_argument('--feature', type=str, default=ftr, help='feature type, [profile, spacy, bert, content]')
                
                args = parser.parse_args()
                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(args.seed)
                
                dataset = FNNDataset(root='data', feature=args.feature, empty=False, name=args.dataset,
                                     transform=DropEdge(args.TDdroprate, args.BUdroprate))
                
                args.num_classes = dataset.num_classes
                args.num_features = dataset.num_features
                
                print(args)
                
                num_training = int(len(dataset) * 0.2)
                num_val = int(len(dataset) * 0.1)
                num_test = len(dataset) - (num_training + num_val)
                training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
                
                # After base run, create new training set excluding graphs in exclude
                if not Net.base and num_splits != 0:
                    current_num_exclude = int(len(exclude)*step/num_splits)
                    training_set = torch.utils.data.Subset(dataset, [i for i in training_set.indices if i not in exclude[:current_num_exclude]])
                    num_training -= int(current_num_exclude)
                
                if args.multi_gpu:
                    loader = DataListLoader
                else:
                    loader = DataLoader
                
                train_loader = loader(training_set, batch_size=args.batch_size, shuffle=True)
                val_loader = loader(validation_set, batch_size=args.batch_size, shuffle=False)
                test_loader = loader(test_set, batch_size=args.batch_size, shuffle=False)
                
                # Create extra train loader with shuffle off, which is passed through compute_test
                # to match embeddings and predictions with graph labels since the order won't change
                train_loader_2 = loader(training_set, batch_size=args.batch_size, shuffle=False)
                
                model = Net(args.num_features, args.nhid, args.nhid)
                if args.multi_gpu:
                    model = DataParallel(model)
                model = model.to(args.device)
                
                if not args.multi_gpu:
                    BU_params = list(map(id, model.BUrumorGCN.conv1.parameters()))
                    BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
                    base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
                    optimizer = torch.optim.Adam([
                        {'params': base_params},
                        {'params': model.BUrumorGCN.conv1.parameters(), 'lr': args.lr / 5},
                        {'params': model.BUrumorGCN.conv2.parameters(), 'lr': args.lr / 5}
                    ], lr=args.lr, weight_decay=args.weight_decay)
                else:
                    BU_params = list(map(id, model.module.BUrumorGCN.conv1.parameters()))
                    BU_params += list(map(id, model.module.BUrumorGCN.conv2.parameters()))
                    base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
                    optimizer = torch.optim.Adam([
                        {'params': base_params},
                        {'params': model.module.BUrumorGCN.conv1.parameters(), 'lr': args.lr / 5},
                        {'params': model.module.BUrumorGCN.conv2.parameters(), 'lr': args.lr / 5}
                    ], lr=args.lr, weight_decay=args.weight_decay)
                
                model.train()
                
                # Early stopping
                # patience: number of epochs without improvement before stopping
                # wait: number of epochs before starting to increase no_improve_count
                # es_epoch: Epoch number where early stopping happens
                patience = 15
                wait = int(epochs/2)
                es_epoch = 0
                best_loss = float('inf')
                no_improve_count = 0
                for epoch in tqdm(range(args.epochs)):
                    out_log = []
                    loss_train = 0.0
                    for i, data in enumerate(train_loader):
                        optimizer.zero_grad()
                        if not args.multi_gpu:
                            data = data.to(args.device)
                        out = model(data)
                        if args.multi_gpu:
                            y = torch.cat([d.y for d in data]).to(out.device)
                        else:
                            y = data.y
                        loss = F.nll_loss(out, y)
                        loss.backward()
                        optimizer.step()
                        loss_train += loss.item()
                        out_log.append([F.softmax(out, dim=1), y])
                        
                    # Added this so the loss graphs make more sense
                    loss_train /= (len(train_loader))
                    
                    acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
                    [acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader)
                    
                    # Record model state to regress back to for early stopping
                    models[epoch] = cp.deepcopy(model.state_dict())
                    # Record accuracy and losses
                    val_losses[epoch] = loss_val
                    train_losses[epoch] = loss_train
                    val_accuracies[epoch] = acc_val
                    
                    # Pass extra train loader through to get embeddings and such from forward()
                    [_, _, _, _, _, _, _], _ = compute_test(train_loader_2)
                    
                    print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
                          f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
                          f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
                          f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')
                    
                    # Early stopping based on validation loss
                    if early_stopping:
                        if loss_val >= best_loss:
                            if no_improve_count < patience:
                                if epoch > wait:
                                    no_improve_count += 1
                            else:
                                print(f'stopped early at: {epoch+1} epochs')
                                break
                        else:
                            no_improve_count = 0
                            best_loss = loss_val
                            es_epoch = epoch + 1
                            
                # If using early stopping load model from best epoch
                if early_stopping:
                    print(f'best epoch: {es_epoch}')
                    model.load_state_dict(models[es_epoch-1])
                else:
                    es_epoch = epochs
                    
                [acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = compute_test(test_loader, verbose=False)
                print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f},'
                      f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')
                
                # Record performances
                if Net.base:
                    base_performances.append([acc, f1_macro])
                elif step == num_splits:
                        full_excl_performances.append([acc, f1_macro])
                seed_excl_performances.append([acc, f1_macro])
                
                step += 1
            
                print(f'train: {num_training} val: {num_val} test: {num_test}')
                
                train_passes = int((num_training - 1)/ args.batch_size) + 1
                val_passes = int((num_val - 1) / args.batch_size) + 1
                test_passes = int((num_test - 1) / args.batch_size) + 1
                total_passes = train_passes + val_passes + test_passes
                
                print(train_passes, val_passes, test_passes)
                
                # Number of passes through the model (calls to forward()) needed for each subset
                train_passes = int((num_training - 1)/ args.batch_size) + 1
                val_passes = int((num_val - 1) / args.batch_size) + 1
                test_passes = int((num_test - 1) / args.batch_size) + 1
                
                # Passes for each epoch. train, validate, plus compute_test for unshuffled training set
                total_passes = 2*train_passes + val_passes
                
                # Get last epoch's embeddings and predictions (w/o test)
                embeddings = Net.embeddings[total_passes*(es_epoch-1):total_passes*(es_epoch)]
                pred_probs = Net.pred_probs[total_passes*(es_epoch-1):total_passes*(es_epoch)]
                
                train_emb = np.concatenate(embeddings[-train_passes:])
                train_pred_probs = np.concatenate(pred_probs[-train_passes:])
                train_pred = np.array([0 if p < 0.5 else 1 for p in train_pred_probs])
                train_graph_labels = np.array(training_set.indices)
                
                # (Test only happens once per run so those are 2D instead of 3D)
                test_emb = np.array(np.concatenate(Net.embeddings[-test_passes:]))
                test_pred_probs = np.concatenate(Net.pred_probs[-test_passes:])
                test_pred = np.array([0 if p < 0.5 else 1 for p in test_pred_probs])
                test_graph_labels = np.array(test_set.indices)
                
                # After the base run get exclude if needed, and save emb and such for slices
                if Net.base:
                    # Get graphs to remove
                    if num_splits != 0:
                        exclude = run_helper.get_exclude(ds, train_emb, train_pred, train_graph_labels, num_exclude, title=f'{ds} bigcn {ftr} {es_epoch} epochs')
                        if num_splits == -1:
                            num_splits = len(exclude)
                        full_excl_num.append(len(exclude))
                    
                    if save_for_slices:
                        os.makedirs('last_run', exist_ok=True)
                        np.save('last_run/pred_probs.npy', test_pred_probs)
                        np.save('last_run/emb.npy', test_emb)
                        np.save('last_run/graph_labels.npy', test_graph_labels)
                        
                        settings = {'dataset':ds, 'model':'bigcn', 'feature':ftr, 'seed':seed, 'epochs':es_epoch}
                        np.save('last_run/settings', settings)
                        
                    Net.base = False
                    
                # Plot pca of test embeddings to see what it looks like
                _ = run_helper.get_exclude(ds, test_emb, test_pred, test_graph_labels, 0, title=f'{ds} bigcn {ftr} test', save_graph=False)
                
                # Reset for next seed
                Net.embeddings = []
                Net.pred = []
                Net.pred_probs = []
                
                # Graph train loss, val loss, and val accuracy
                plt.plot(range(1, epochs+1), val_losses)
                plt.plot(range(1, epochs+1), train_losses)
                plt.plot(range(1, epochs+1), val_accuracies)
                plt.vlines(es_epoch, min(val_accuracies+val_losses+train_losses), max(val_accuracies+val_losses+train_losses), colors=['k'])
                plt.title(f'{args.dataset} bigcn {args.feature}')
                plt.legend(["val loss", "train loss", "val accuracy"])
                plt.xlabel('epochs')
                
                plt.savefig('last_run/model_training.png', dpi=1200)
                
                plt.show()
                
            # Plot performances with modified training set vs base performance
            if num_splits != 0:
                excl_acc = [p[0] for p in seed_excl_performances]
                excl_f1 = [p[1] for p in seed_excl_performances]
                num_removed = [int(len(exclude)*s/num_splits) for s in range(num_splits+1)]
                
                plt.plot(num_removed, excl_acc, label='accuracy')
                plt.plot(num_removed, excl_f1, label='f1')
                plt.axhline(base_performances[-1][0], linestyle='--', color='green', label='base acc')
                plt.axhline(base_performances[-1][1], linestyle='--', color='red', label='base f1')
                plt.xlabel('Graphs removed')
                plt.xticks(ticks=num_removed, labels=num_removed)
                plt.legend()
                
                plt.savefig('last_run/removal_performance.png', dpi=1200)
                
                plt.show()
                
                # Track performance and number of graphs removed for the best removal run (highest f1)
                best_excl_perf, best_num = max([(p, num) for p, num in zip(seed_excl_performances, num_removed)], key=lambda x: x[0][1])
                best_excl_performances.append(best_excl_perf)
                best_excl_num.append(best_num)        
                
        # Get average performances
        base_acc_avg = np.mean([acc for acc, f1 in base_performances])
        base_f1_avg = np.mean([f1 for acc, f1 in base_performances])
        
        if num_splits != 0:
            excl_acc_avg = np.mean([acc for acc, f1 in full_excl_performances])
            excl_f1_avg = np.mean([f1 for acc, f1 in full_excl_performances])
            
            best_acc_avg = np.mean([acc for acc, f1 in best_excl_performances])
            best_f1_avg = np.mean([f1 for acc, f1 in best_excl_performances])
        
        # Print and log average performances and those for each seed
        with open('performance_log.txt' ,'a') as outfile:
            print(f'\nbase avg: {base_acc_avg:.4f}, {base_f1_avg:.4f}')
            outfile.write(f'{ds} bigcn {ftr}\n')
            outfile.write(f'base avg: {base_acc_avg:.4f}, {base_f1_avg:.4f}\n')
            for i in range(len(seeds)):
                print(f'seed {seeds[i]}: {base_performances[i][0]:.4f}, {base_performances[i][1]:.4f}')
                outfile.write(f'seed {seeds[i]}: {base_performances[i][0]:.4f}, {base_performances[i][1]:.4f}\n')
                if num_splits != 0:
                    print(f'\tfull exclude: {full_excl_num[i]} {full_excl_performances[i][0]:.4f}, {full_excl_performances[i][1]:.4f}')
                    print(f'\tbest exclude: {best_excl_num[i]} {best_excl_performances[i][0]:.4f}, {best_excl_performances[i][1]:.4f}')
                    outfile.write(f'\tfull exclude: {full_excl_num[i]} {full_excl_performances[i][0]:.4f}, {full_excl_performances[i][1]:.4f}\n')
                    outfile.write(f'\tbest exclude: {best_excl_num[i]} {best_excl_performances[i][0]:.4f}, {best_excl_performances[i][1]:.4f}\n')
            if num_splits != 0:
                print(f'full excl avg: {excl_acc_avg:.4f}, {excl_f1_avg:.4f}')
                print(f'best excl avg: {best_acc_avg:.4f}, {best_f1_avg:.4f}')
                outfile.write(f'full excl avg: {excl_acc_avg:.4f}, {excl_f1_avg:.4f}\n')
                outfile.write(f'best excl avg: {best_acc_avg:.4f}, {best_f1_avg:.4f}\n')
            outfile.write('\n')
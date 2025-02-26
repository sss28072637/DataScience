from argparse import ArgumentParser

from data_loader import load_data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from model import Model, Encoder, drop_feature # Build your model in model.py
# from model import Net_orig # Build your model in model.py
    
import os
import warnings

from torch_geometric.utils import from_networkx
import dgl

from torch_geometric.utils import dropout_adj

warnings.filterwarnings("ignore")

# def evaluate(g, features, labels, mask, model):
#     """Evaluate model accuracy"""
#     model.eval()
#     with torch.no_grad():
#         logits = model(g, features, mask)
#         logits = logits[mask]
#         _, indices = torch.max(logits, dim=1)
#         correct = torch.sum(indices == labels)
#         return correct.item() * 1.0 / len(labels)

# def train(g, features, train_labels, val_labels, train_mask, val_mask, model, epochs, es_iters=None):
    
#     # define train/val samples, loss function and optimizer
#     loss_fcn = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

#     # If early stopping criteria, initialize relevant parameters
#     if es_iters:
#         print("Early stopping monitoring on")
#         loss_min = 1e8
#         es_i = 0

#     # training loop
#     for epoch in range(epochs):
#         model.train()
#         # logits = model(g, features)
#         logits = model(g, features, train_mask)
#         loss = loss_fcn(logits[train_mask], train_labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         acc = evaluate(g, features, val_labels, val_mask, model)
#         print(
#             "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
#                 epoch, loss.item(), acc
#             )
#         )
        
#         val_loss = loss_fcn(logits[val_mask], val_labels).item()
#         if es_iters:
#             if val_loss < loss_min:
#                 loss_min = val_loss
#                 es_i = 0
#             else:
#                 es_i += 1

#             if es_i >= es_iters:
#                 print(f"Early stopping at epoch={epoch+1}")
#                 break

def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)

    label_classification(z, y, ratio=0.1)


if __name__ == '__main__':

    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--es_iters', type=int, help='num of iters to trigger early stopping')
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    if args.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Load data
    features, graph, num_classes, \
    train_labels, val_labels, test_labels, \
    train_mask, val_mask, test_mask = load_data()
    
    nx_graph = dgl.to_networkx(graph)
    pyg_graph = from_networkx(nx_graph)
    graph = pyg_graph
    
    features = features.to(device)
    train_labels = train_labels.to(device)
    val_labels = val_labels.to(device)
    test_labels = torch.tensor(test_labels).to(device)
    train_mask = torch.tensor(train_mask).to(device)
    val_mask = torch.tensor(val_mask).to(device)
    test_mask = torch.tensor(test_mask).to(device)
    graph = graph.to(device)


    learning_rate = 0.0005
    num_hidden = 128
    num_proj_hidden = 128
    activation = F.relu
    base_model = GCNConv
    num_layers = 2

    drop_edge_rate_1 = 0.2
    drop_edge_rate_2 = 0.4
    drop_feature_rate_1 = 0.3
    drop_feature_rate_2 = 0.4
    tau = 0.4
    num_epochs = 200
    weight_decay = 0.00001
    
    # Initialize the model (Baseline Model: GCN)
    """TODO: build your own model in model.py and replace GCN() with your model"""
    in_size = features.shape[1]
    out_size = num_classes
    # model = GCN(in_size, 16, out_size).to(device)
    # model = YourGNNModel(in_size, 16, out_size).to(device)
    # model = Net(features, out_size, 16, 0.5).to(device)
    # model = Net_orig(16, in_size, out_size).to(device)

    encoder = Encoder(in_size, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    print("Training...")

    for epoch in range(1, num_epochs + 1):
        loss = train(model, features[train_mask]+features[val_mask], graph.edge_index)
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
    
    # model training
    # train(graph, features, train_labels, val_labels, train_mask, val_mask, model, args.epochs, args.es_iters)
    
    print("Testing...")
    # model.eval()
    # with torch.no_grad():
    #     logits = model(graph, features, test_mask)
    #     logits = logits[test_mask]
    #     _, indices = torch.max(logits, dim=1)
    
    # Export predictions as csv file
    print("Export predictions as csv file.")
    with open('output.csv', 'w') as f:
        f.write('ID,Predict\n')
        for idx, pred in enumerate(indices):
            f.write(f'{idx},{int(pred)}\n')
    # Please remember to upload your output.csv file to Kaggle for scoring
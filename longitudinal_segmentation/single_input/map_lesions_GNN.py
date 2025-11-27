"""
This code trains a Graph Neural Network (GNN) to track lesions over time in longitudinal medical imaging data.
It takes as input the csv dataset and trains a model.

Input:
    - dataset_csv: Path to the csv dataset containing lesion features over time.
    - output_folder: Path to the folder where model and results will be saved.

Output:
    None

Author: Pierre-Louis Benveniste
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MetaLayer
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import networkx as nx
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-csv', type=str, required=True, help='Path to the csv dataset containing lesion features over time')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to the output folder where model and results will be saved')
    return parser.parse_args()


def distance_cylindrical(coord1, coord2, w_z=1.0, w_disk=1.0):
    """
    Computes a weighted Euclidean distance between two points in cylindrical coordinates.
    
    Parameters:
        coord1 (dict): Coordinates of the first point with keys 'r', 'theta', and 'z'.
        coord2 (dict): Coordinates of the second point with keys 'r', 'theta', and 'z'.
        w_z (float): Weight for the z-axis distance.
        w_disk (float): Weight for the distance in the disk plane (r, theta).
    
    Returns:
        float: The weighted Euclidean distance.
    """
    distance_disk = np.sqrt(coord1['r']**2 + coord2['r']**2 - 2 * coord1['r'] * coord2['r'] * np.cos(np.radians(coord1['theta'] - coord2['theta'])))
    z_dist = coord1['z'] - coord2['z']
    return np.sqrt(w_z * z_dist**2 + w_disk * distance_disk**2)


# ==========================================
# PART 1: Graph Construction (The Hard Part)
# ==========================================
def create_graph_from_subject(df_subject, max_dist=50.0):
    """
    Converts a pandas DataFrame of 1 subject into a PyG Data object.
    Each row in df_subject is a lesion at a specific timepoint with 
        'subject': subject,
        'timepoint': timepoint1,
        'group': lesion_info['group'],
        'z': lesion_info['centerline_z'],
        'r': lesion_info['radius_mm'],
        'theta': lesion_info['theta'],
        'volume': lesion_info['volume_mm3']
    """

    # 1. Node Features: [z, r, theta, volume, timepoint]
    node_features = torch.tensor(
        df_subject[['z', 'r', 'theta', 'volume', 'timepoint']].values, 
        dtype=torch.float
    )
    
    # 2. Create Candidate Edges
    # We only want edges between:
    #   a) Time T and Time T+1 (Tracking)
    #   b) Time T and Time T   (Merging/Segmentation artifacts)
    
    times = df_subject['time_point'].values
    coords = df_subject[['x', 'y', 'z']].values
    num_nodes = len(df_subject)
    
    edge_sources = []
    edge_targets = []
    edge_attrs = []
    
    # Naive generic loop (O(N^2)) - for production use scipy.spatial.KDTree
    for i in range(num_nodes):
        for j in range(num_nodes):
            # No self-loops
            if i == j: continue
            # No identical time
            if times[i] == times[j]: continue
            # No backward time (since we have only 2 timepoints, ses-M12 has to be j)
            if times[i] == "ses-M12": continue
                
            # Constraint 2: Distance threshold (in cylindrical coordinates)
            dist = distance_cylindrical(coords[i], coords[j])
            # If too far, skip (commenting for now)
            # if dist > max_dist:
            #     continue
            
            # Create Edge
            edge_sources.append(i)
            edge_targets.append(j)
            
            # Edge Features: [distance, vol_diff, direction_x, direction_y, direction_z]
            vol_diff = abs(df_subject.iloc[i]['vol'] - df_subject.iloc[j]['vol'])
            direction = coords[j] - coords[i]
            edge_attrs.append([dist, vol_diff, direction[0], direction[1], direction[2]])

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    # 3. Create Ground Truth Labels (y) for Edges
    # If we have ground truth 'track_id', we set y=1 if they share the same track_id
    y_edges = []
    track_ids = df_subject['track_id'].values
    for k in range(len(edge_sources)):
        u, v = edge_sources[k], edge_targets[k]
        # Match if same track_id
        label = 1.0 if track_ids[u] == track_ids[v] else 0.0
        y_edges.append(label)
    y = torch.tensor(y_edges, dtype=torch.float)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)





###############
#############
#### J'EN ETAIS LA 
#############
###############







# ==========================================
# PART 2: The Model (Link Prediction GNN)
# ==========================================
class EdgeEncoder(nn.Module):
    def __init__(self, node_in, edge_in, hidden):
        super().__init__()
        # MLP to look at Node_i, Node_j, and Edge_ij and decide if they match
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_in + edge_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1) # Output: Score (Logits)
        )

    def forward(self, src, dst, edge_attr, u=None, batch=None):
        # src, dst: [E, node_in]
        # edge_attr: [E, edge_in]
        out = torch.cat([src, dst, edge_attr], dim=1)
        return self.mlp(out)

class LesionTrackerGNN(nn.Module):
    def __init__(self, node_dim=5, edge_dim=5, hidden_dim=64):
        super().__init__()
        # We use a MetaLayer for flexible edge/node updates
        self.edge_model = EdgeEncoder(node_dim, edge_dim, hidden_dim)

    def forward(self, data):
        # Returns raw logits for every edge
        # data.x needs to be mapped to src and dst for every edge
        row, col = data.edge_index
        
        # Helper to get features of source and target nodes for each edge
        src_features = data.x[row]
        dst_features = data.x[col]
        
        edge_scores = self.edge_model(src_features, dst_features, data.edge_attr)
        return edge_scores.squeeze()

# ==========================================
# PART 3: Grouping Logic (Union-Find)
# ==========================================
def predict_and_group(model, data, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits)
    
    # 1. Filter edges: Keep only those with prob > threshold
    active_edges_mask = probs > threshold
    active_edges = data.edge_index[:, active_edges_mask].cpu().numpy()
    
    # 2. Build a graph of ONLY the active edges
    # We use NetworkX because it has a fast "connected_components" implementation
    # This IS the Union-Find logic you asked for.
    G = nx.Graph()
    num_nodes = data.x.shape[0]
    G.add_nodes_from(range(num_nodes)) # Add all nodes (even isolated ones)
    
    # Add predicted edges
    edge_list = list(zip(active_edges[0], active_edges[1]))
    G.add_edges_from(edge_list)
    
    # 3. Get Groups (Connected Components)
    groups = list(nx.connected_components(G))
    
    # 4. Format Output
    # Create a mapping: Node_Index -> Group_ID
    node_to_group = {}
    for group_id, group_nodes in enumerate(groups):
        for node in group_nodes:
            node_to_group[node] = group_id
            
    return node_to_group

# # ==========================================
# # PART 4: Example Usage
# # ==========================================

# # --- A. Create Fake Data ---
# data_dict = {
#     'lesion_id': [0, 1, 2, 3, 4],
#     'x': [10, 12, 50, 11, 48],
#     'y': [10, 12, 50, 11, 49],
#     'z': [10, 10, 50, 10, 51],
#     'vol': [100, 110, 500, 105, 510],
#     'time_point': [0, 1, 0, 1, 1],
#     'track_id':   [1, 1, 2, 1, 2] # Ground Truth: 0->1->3 are same, 2->4 are same
# }
# df = pd.DataFrame(data_dict)

# # --- B. Setup ---
# # Create Graph
# graph_data = create_graph_from_subject(df)
# # Initialize Model
# model = LesionTrackerGNN()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = nn.BCEWithLogitsLoss() # Good for binary edge classification

# # --- C. Training Loop (Standard) ---
# model.train()
# for epoch in range(100):
#     optimizer.zero_grad()
    
#     out = model(graph_data) # Output: logits for edges
    
#     # Calculate loss only on known edges
#     loss = criterion(out, graph_data.y)
    
#     loss.backward()
#     optimizer.step()
    
#     if epoch % 20 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# # --- D. Final Tracking / Grouping ---
# # This runs the Union-Find logic on the model's predictions
# final_grouping = predict_and_group(model, graph_data)

# print("\n--- Final Results ---")
# print("Node ID | Predicted Group | True Track ID")
# for i in range(len(df)):
#     print(f"   {i}    |        {final_grouping[i]}        |       {df.iloc[i]['track_id']}")


def train():
    args = parse_args()
    dataset_csv = args.dataset_csv
    output_folder = args.output_folder

    # Load dataset
    df = pd.read_csv(dataset_csv)

    # Replace column 'group' to 'track_id' for clarity
    df = df.rename(columns={'group': 'track_id'})

    # Create graphs for each subject
    subjects = df['subject_id'].unique()
    graphs = []
    for subject in subjects:
        df_subject = df[df['subject_id'] == subject]
        graph = create_graph_from_subject(df_subject)
        graphs.append(graph)

    # # Create DataLoader
    # loader = DataLoader(graphs, batch_size=1, shuffle=True)

    # # Initialize model, optimizer, loss
    # model = LesionTrackerGNN()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.BCEWithLogitsLoss()

    # # Training loop
    # num_epochs = 50
    # for epoch in range(num_epochs):
    #     model.train()
    #     total_loss = 0
    #     for data in loader:
    #         optimizer.zero_grad()
    #         out = model(data)
    #         loss = criterion(out, data.y)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #     avg_loss = total_loss / len(loader)
    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # # Save model
    # os.makedirs(output_folder, exist_ok=True)
    # model_path = os.path.join(output_folder, 'lesion_tracker_gnn.pth')
    # torch.save(model.state_dict(), model_path)
    # print(f"Model saved to {model_path}")
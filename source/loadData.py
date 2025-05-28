import gzip
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm 
import json

class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None, is_zipped=True):
        self.raw = filename
        self.graphs = self.loadGraphs(self.raw, is_zipped)
        super().__init__(None, transform, pre_transform)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    @staticmethod
    def loadGraphs(path, is_zipped):
        print(f"Loading graphs from {path}...")
        print("This may take a few minutes, please wait...")
        if is_zipped:
            with gzip.open(path, "rb") as f:
                graphs_dicts = json.loads(f.read())
        else:
            with open(path, "rb") as f:
                graphs_dicts = json.loads(f.read())
        graphs = []
        print("json loaded, processing the data...")
        for graph_dict in tqdm(graphs_dicts, desc="Processing graphs", unit="graph"):
            graphs.append(dictToGraphObject(graph_dict))
        return graphs



def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
    num_nodes = graph_dict["num_nodes"]
    y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)
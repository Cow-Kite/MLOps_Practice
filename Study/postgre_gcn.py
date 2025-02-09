import time
import torch
import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
# mlflow
import mlflow
import mlflow.pytorch
import uuid

run_name = f"GCN_Run_{uuid.uuid4().hex[:8]}"

MLFLOW_TRACKING_URI = "postgresql://mlflow:mlflow@localhost:5432/mlflow"

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        output = self.conv2(x, edge_index)
        return output
    

def train_node_classifier(model, graph, optimizer, criterion, n_epochs=200):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("epochs", n_epochs)
        mlflow.log_param("learning_rate", optimizer.param_groups[0]['lr'])
        mlflow.log_param("weight_decay", optimizer.param_groups[0]['weight_decay'])

        start_time = time.time()

        for epoch in range(1, n_epochs + 1):
            model.train() 
            optimizer.zero_grad() 
            out = model(graph) 
            loss = criterion(out[graph.train_mask], graph.y[graph.train_mask]) 
            loss.backward() 
            optimizer.step() 

            pred = out.argmax(dim=1) 
            acc = eval_node_classifier(model, graph, graph.val_mask)

            mlflow.log_metric("train_loss", loss.item(), step=epoch)
            mlflow.log_metric("val_acc", acc, step=epoch)

            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')

        total_time = time.time() - start_time
        mlflow.log_metric("training_time", total_time)
        print(f"Training completed in {total_time:.2f} seconds.")

        mlflow.pytorch.log_model(model, "trained_gcn")

        test_acc = eval_node_classifier(model, graph, graph.test_mask)
        mlflow.log_metric("test_acc", test_acc) 

        print(f"Test Accuracy: {test_acc:.3f}")
        return model

def eval_node_classifier(model, graph, mask):
    model.eval() 
    pred = model(graph).argmax(dim=1)
    correct = (pred[mask] == graph.y[mask]).sum()
    
    acc = int(correct) / int(mask.sum())
    return acc


if __name__=="__main__":
    # set mlflow
    study_name="GCN-tutorial"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(study_name)

    device = "cpu"

    dataset = Planetoid(root='./', name='Cora')
    graph = dataset[0]
    split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
    graph = split(graph)

    gcn = GCN().to(device)
    optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    gcn = train_node_classifier(gcn, graph, optimizer_gcn, criterion)


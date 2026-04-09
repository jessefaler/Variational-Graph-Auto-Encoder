import time
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_dense_adj

from vgae import VGAE, encode_mean, symmetric_normalized_adj, train_vgae

def link_pred_metrics(z, edge_label_index, edge_label):
    # Score pairs of nodes for link prediction and compute AUC and AP metrics
    row, col = edge_label_index[0], edge_label_index[1]
    scores = (z[row] * z[col]).sum(dim=1)
    y_true = edge_label.cpu().numpy()
    y_score = scores.detach().cpu().numpy()
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return auc, ap


def main():
    # Set the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Cora dataset with features and edges
    dataset = Planetoid(root="./data", name="Cora")
    data = dataset[0].to(device)

    # Split edges for training, validation, and test
    splitter = RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = splitter(data)

    # Extract node features and number of nodes
    x = train_data.x
    n = x.size(0)

    # Build dense adjacency matrices for training and decoder target
    train_edges = train_data.edge_index
    a_train = to_dense_adj(train_edges, max_num_nodes=n)[0].float()
    adj_label = a_train + torch.eye(n, device=device, dtype=a_train.dtype)
    a_hat = symmetric_normalized_adj(a_train)

    # Initialize model
    hidden_dim = 32
    latent_dim = 16
    model = VGAE(
        input_dim=x.size(1),
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)

    # Set training parameters
    steps = 200
    lr = 0.01

    # Train the model and measure training time
    start = time.perf_counter()
    train_vgae(model, x, adj_label, a_train, a_hat, steps=steps, lr=lr)
    elapsed = time.perf_counter() - start

    # Compute deterministic embeddings for evaluation
    mean = encode_mean(model, x, a_hat)

    # Evaluate link prediction on validation and test sets
    val_auc, val_ap = link_pred_metrics(mean, val_data.edge_label_index, val_data.edge_label)
    test_auc, test_ap = link_pred_metrics(mean, test_data.edge_label_index, test_data.edge_label)

    print("\ndataset:", dataset.name)
    print("train steps:", steps, "learning rate:", lr)
    print("hidden_dim:", hidden_dim, "latent_dim:", latent_dim)
    print(f"training_time: {round(elapsed, 4)}s")
    print("validation AUC:", round(val_auc, 4), "AP:", round(val_ap, 4))
    print("test AUC:", round(test_auc, 4), "AP:", round(test_ap, 4))

if __name__ == "__main__":
    main()
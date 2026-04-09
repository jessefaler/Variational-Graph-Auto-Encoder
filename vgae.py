import torch
import torch.nn as nn
import torch.nn.functional as F


def symmetric_normalized_adj(adj):
    # Compute the symmetrically normalized adjacency matrix with self loops.
    n = adj.size(0)
    I = torch.eye(n, device=adj.device, dtype=adj.dtype)
    adj = adj + I  # add self loops

    deg = adj.sum(dim=1).clamp(min=1e-12)
    inv_sqrt = torch.pow(deg, -0.5)
    D = torch.diag(inv_sqrt)

    return D @ adj @ D

class GCNLayer(nn.Module):
    # One graph convolution layer that propagates features and applies a linear transform
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, adj_hat):
        # Graph propagation then linear
        return self.linear(adj_hat @ x)

class VGAE(nn.Module):
    # Variational Graph Auto Encoder with two layer GCN encoder.
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.hidden_layer = GCNLayer(input_dim, hidden_dim)
        self.mean_layer = GCNLayer(hidden_dim, latent_dim)
        self.logvar_layer = GCNLayer(hidden_dim, latent_dim)

    def encode(self, x, adj_hat):
        # Encode node features into mean and log variance.
        hidden = F.relu(self.hidden_layer(x, adj_hat))
        mean = self.mean_layer(hidden, adj_hat)
        logvar = self.logvar_layer(hidden, adj_hat)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        # z = mean + eps * sigma
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        # Reconstruct the adjacency matrix using the inner product decoder.
        return z @ z.t()

    def forward(self, x, a_hat):
        mean, logvar = self.encode(x, a_hat)
        z = self.reparameterize(mean, logvar)
        logits = self.decode(z)
        return logits, mean, logvar

def elbo_loss(recon_logits, adj_target, mean, logvar, adj_train):
    # Compute the ELBO loss by reconstruction and KL divergence.
    num_nodes = mean.size(0)
    num_elements = adj_target.numel()

    # Weight positive edges for sparse graphs
    pos_weight = (num_elements - float(adj_train.sum().item())) / float(adj_train.sum().item())
    norm = num_elements / (2.0 * (num_elements - float(adj_train.sum().item())))

    recon_loss = norm * F.binary_cross_entropy_with_logits(
        recon_logits, adj_target, pos_weight=torch.tensor(pos_weight, device=recon_logits.device)
    )

    # Compute the KL divergence between each node’s learned latent distribution and a standard normal distribution.
    kl_loss = 0.5 / num_nodes * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=1).mean()

    return recon_loss - kl_loss

def train_vgae(model, x, adj_target, adj_train, adj_hat, steps, lr):
    # Train the VGAE model on the full graph.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(steps):
        optimizer.zero_grad()
        recon_logits, mean, logvar = model(x, adj_hat)
        loss = elbo_loss(recon_logits, adj_target, mean, logvar, adj_train)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

    return model

@torch.no_grad()
def encode_mean(model, x, adj_hat):
    model.eval()
    mean, _ = model.encode(x, adj_hat)
    return mean
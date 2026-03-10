import torch
import torch.nn as nn

class mlp_pod_trainer:
    def __init__(self, model, dataloader, xgrid1d_ten, xgrid1d_bc_ten, epochs=40, lr=0.005, update_interval=5):
        self.model = model
        self.dataloader = dataloader
        self.xgrid1d_ten = xgrid1d_ten
        self.xgrid1d_bc_ten = xgrid1d_bc_ten
        self.epochs = epochs
        self.update_interval = update_interval
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.w_data = 1.0
        self.w_bc = 1.0

    def _compute_dynamic_weights(self):
        """Recompute loss components and update weights dynamically (data and bc only)."""
        self.model.eval()
        with torch.no_grad():
            ti, xi, ui = next(iter(self.dataloader))  # One batch
            outputs = self.model(ti, xi)
            loss_data = self.loss_fn(outputs, ui).item()

            bc_values = self.model.basis_net(self.xgrid1d_bc_ten)
            loss_bc = (bc_values[0] - bc_values[1]).pow(2).item()

            # Inverse losses as weights, normalized
            weights = torch.tensor([loss_data, loss_bc]) + 1e-8
            inv_weights = 1.0 / weights
            norm_weights = inv_weights / inv_weights.sum()

            self.w_data, self.w_bc = norm_weights.tolist()

    def train(self):
        for epoch in range(self.epochs):
            if epoch % self.update_interval == 0:
                self._compute_dynamic_weights()

            self.model.train()
            total_loss = 0.0

            for ti, xi, ui in self.dataloader:
                self.optimizer.zero_grad()

                outputs = self.model(ti, xi)
                loss_data = self.loss_fn(outputs, ui)

                bc_values = self.model.basis_net(self.xgrid1d_bc_ten)
                loss_bc = (bc_values[0] - bc_values[1]).pow(2).item()

                # Weighted total loss (no basis normalization)
                total = self.w_data * loss_data + self.w_bc * loss_bc

                total.backward()
                self.optimizer.step()
                total_loss += total.item()

            if epoch % 2 == 0:
                print(f"Epoch {epoch} | Loss: {total_loss / len(self.dataloader):.6f} | "
                      f"Weights: data={self.w_data:.3f}, bc={self.w_bc:.3f}")

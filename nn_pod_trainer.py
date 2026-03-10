import torch
import torch.nn as nn

class nn_pod_trainer:
    def __init__(self, model, dataloader, xgrid1d_ten, xgrid1d_bc_ten, epochs=40, lr=0.005):
        """
        Args:
            model (nn.Module): The model to train.
            dataloader (DataLoader): PyTorch dataloader providing training batches.
            xgrid1d_tensor (torch.Tensor): The grid tensor for computing the basis penalty.
            epochs (int): Number of training epochs.
            lr (float): Learning rate for the optimizer.
        """
       self.model = model
        self.dataloader = dataloader
        self.xgrid1d_ten = xgrid1d_ten
        self.xgrid1d_bc_ten = xgrid1d_bc_ten
        self.epochs = epochs
        
        # Define optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # ---- 1) Compute initial losses (on a single batch) ----
        self.model.eval()
        ti_init, xi_init, ui_init = next(iter(self.dataloader))  # First batch
        with torch.no_grad():
            outputs_init = self.model(ti_init, xi_init)
            L_data_init = self.loss_fn(outputs_init, ui_init).item()

            basis_values_init = self.model.basis_net(self.xgrid1d_tensor)
            N_init = basis_values_init.numel()
            L_basis_init = (
                torch.abs(torch.sqrt(torch.sum(basis_values_init**2) / N_init) - 1)
            ) ** 2
            L_basis_init = L_basis_init.item()

        # Compute reciprocal weights for each loss to balance training
        self.w_data = 1.0 / (L_data_init + 1e-8)
        self.w_basis = 1.0 / (L_basis_init + 1e-8)

    def train(self):
        """Run the training loop for the specified number of epochs."""
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for ti, xi, ui in self.dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(ti, xi)
                
                # Data loss
                loss_data = self.loss_fn(outputs, ui)
                
                # Basis penalty
                basis_values = self.model.basis_net(self.xgrid1d_tensor)
                N = basis_values.numel()
                loss_basis_rms = (
                    torch.abs(torch.sqrt(torch.sum(basis_values**2) / N) - 1)
                ) ** 2
                
                # Weighted loss
                loss = self.w_data * loss_data + self.w_basis * loss_basis_rms
                
                # Backprop and update
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # Print every 2 epochs
            if epoch % 2 == 0:
                print(f"Epoch {epoch} | Loss: {total_loss / len(self.dataloader):.6f}")

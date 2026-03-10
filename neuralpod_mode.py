import numpy as np
import matplotlib.pyplot as plt
import torch

class neuralpod_mode:
    def __init__(self, model, xgrid1d_ten, x, vs, pod_index=1, device='cpu'):
        """
        Args:
            model (torch.nn.Module): Trained model with a .basis_net method.
            xgrid1d_ten (torch.Tensor): Grid tensor of shape [N, 1].
            x (np.ndarray): Grid values for plotting (should match xgrid1d_ten.squeeze()).
            vs (np.ndarray): POD modes (shape: [N, num_modes]).
            pod_index (int): Index of POD mode to compare (default: 1 for second mode).
            device (str): Torch device.
        """
        self.model = model
        self.xgrid1d_ten = xgrid1d_ten.to(device)
        self.x = x
        self.vs = vs
        self.pod_index = pod_index
        self.device = device

        self.phi_kanpod_normalized = None
        self.phi_pod_normalized = None
        self.l2_kanpod = None

    def compute_basis_functions(self):
        """Evaluate and normalize neural and POD basis functions."""
        self.model.eval()
        with torch.no_grad():
            phi_kanpod = self.model.basis_net(self.xgrid1d_ten).view(-1).cpu().numpy()
        
        phi_pod = self.vs[:, self.pod_index]
        xgrid_np = self.xgrid1d_ten.squeeze(dim=1).cpu().numpy()

        # L2 normalization
        l2_kanpod = np.sqrt(np.trapz(phi_kanpod**2, xgrid_np))
        l2_pod = np.sqrt(np.trapz(phi_pod**2, xgrid_np))

        self.phi_kanpod_normalized = phi_kanpod / l2_kanpod
        self.phi_pod_normalized = phi_pod / l2_pod
        self.l2_kanpod = l2_kanpod
        
    def plot(self, filename='basis_comparison.pdf'):
        """Plot and save the comparison figure."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.x, self.phi_kanpod_normalized, 'r', label='Neural POD')
        plt.plot(self.x, self.phi_pod_normalized, 'b', label='POD')
        plt.xlabel('x')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0)
        plt.show()

    def run(self, filename='basis_comparison.pdf'):
        """
        Run the pipeline: compute, normalize, and plot.

        Returns:
            Tuple of (phi_pod_normalized, phi_kanpod_normalized, l2_kanpod)
        """
        self.compute_basis_functions()
        self.plot(filename)
        return self.phi_pod_normalized, self.phi_kanpod_normalized, self.l2_kanpod

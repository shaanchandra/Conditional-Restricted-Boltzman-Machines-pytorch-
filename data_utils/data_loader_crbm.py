import numpy as np
import torch

class CRBMDataset(torch.utils.data.Dataset):
    """CRBM Dataset for sampling data with their respective conditional variables
    Args:
        - vis (numpy.ndarray): the visible variables (D x F)
        - cond (numpy.ndarray): the conditional variables (D x (F_t-1 + F_static))
    Parameters:
        - vis (torch.FloatTensor): the visible variables
        - cond (torch.FloatTensor): the conditional variables
    """
    def __init__(self, vis, cond):
        # sanity check
        if len(vis) != len(cond):
            raise ValueError(f"len(vis) `{len(vis)}` != len(cond) {len(cond)}")
        

        self.vis = torch.FloatTensor(vis)
        self.cond = torch.FloatTensor(cond)

    def __len__(self):
        return len(self.vis)

    def __getitem__(self, idx):
        return self.vis[idx], self.cond[idx]

    def collate_fn(self, batch):
        """Minibatch sampling
        """
        # Pad sequences to max length
        X_vis = [X for X in batch[0]]
        X_cond = [X for X in batch[1]]

        return X_vis, X_cond
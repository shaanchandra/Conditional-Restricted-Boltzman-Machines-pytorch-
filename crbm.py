import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
from sklearn.neighbors import KDTree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class CRBM(nn.Module):
    """Conditional Restricted Boltzmann Machine.
    Args:
        n_vis (int, optional): The size of visible layer.
        n_hid (int, optional): The size of hidden layer.
        k (int, optional): The number of Gibbs sampling.
    """

    def __init__(self, config, k=1):
        """Create a CRBM like Unlearn.ai"""
        super(CRBM, self).__init__()
        self._nv = config['crbmParams']['input_dim']
        self._nh = config['crbmParams']['hidden_dim']
        self._ncond = config['crbmParams']['cond_dim']
        self._batch_size = config['commonModelParameter']['batch_size']
        self.W = nn.Parameter(torch.randn(self._nh, self._nv))
        self.cond_vis = nn.Parameter(torch.randn(self._ncond, self._nv))
        self.cond_hid = nn.Parameter(torch.randn(self._ncond, self._nh))
        self.v = nn.Parameter(torch.randn(1, self._nv))
        self.h = nn.Parameter(torch.randn(1, self._nh))
        self.k = k



    def vc_to_h(self, v, cond):
        """Conditional sampling a hidden variable given a visible variable.
        Args:
            v (Tensor): The visible variable.
            cond (Tensor): The conditional variable.
        Returns:
            p (Tensor): The hidden variable.
        """
        p = torch.sigmoid(F.linear(v, self.W, self.h) + F.linear(cond, self.cond_hid.T, self.h))
        # return p.bernoulli()
        return p


    def hc_to_v(self, h, cond):
        """Conditional sampling a visible variable given a hidden variable.
        Args:
            h (Tendor): The hidden variable.
            cond (Tensor): The conditional variable.
        Returns:
            Tensor: The visible variable.
        """
        p = torch.sigmoid(F.linear(h, self.W.T, self.v) + F.linear(cond, self.cond_vis.T, self.v))
        # return p.bernoulli()
        return p
    

    def gibbs_sample_hvh(self, h_samples0, cond):
        """
        Runs a cycle of gibbs sampling, started with an initial hidden units activations (Used for training)
        Args:
            h_samples0: a tensor of initial hidden units activations
            cond: a tensor of condition units configurations
        Returns:
            v_samples: visible samples
            h_samples: a tensor containing the mean probabilities of the hidden units activations
        """
        # v from h
        v_samples = self.hc_to_v(h_samples0, cond)
        # h from v
        h_samples = self.vc_to_h(v_samples, cond)
        return v_samples, h_samples


    def gibbs_sample_vhv(self, v_samples0, cond):
        """
        Runs a cycle of gibbs sampling, started with an initial visible and condition units
        Args:
            v_samples0: a tensor of initial visible units configuration
            cond: a tensor of condition units configurations
        Returns:
            v_samples: visible samples
            h_samples: a tensor containing the mean probabilities of the hidden units activations
        """
        h_samples = self.vc_to_h(v_samples0, cond) # h from v
        v_samples = self.hc_to_v(h_samples, cond) # v from h
        return v_samples, h_samples



    def free_energy(self, v, cond):
        """Free energy function.
        .. math::
            \begin{align}
                F(x) &= -\log \sum_h \exp (-E(x, h)) \\
                &= -a^\top x - \sum_j \log (1 + \exp(W^{\top}_jx + b_j))\,.
            \end{align}
        Args:
            v (Tensor): The visible variable.
        Returns:
            FloatTensor: The free energy value.
        (https://github.com/taneishi/CRBM/blob/master/crbm.py)
        """
        w_x_h = F.linear(v, self.W, self.h) + F.linear(cond, self.cond_hid, self.h)        
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        a_x_v = F.linear(cond, self.cond_vis, self.v)
        v_term = torch.sum(0.5 * torch.sqrt(v - a_x_v), axis=1)
        return torch.mean(v_term - h_term)
    

    def reconstruction_loss(self, v, v_gibb):
        """
        Reconstruction error is a better proxy for Contrastive Divergence (CD)
        Args:
            v (Tensor): The visible variable (orig).
            v_gibb (Tensor): The visible variable (generated).
            cond (Tensor): the conditional variable
        Returns:
            FloatTensor: The reconstruction loss value.
        """
        recon_loss = (v-v_gibb)**2
        return recon_loss.mean()


    def forward(self, v, cond):
        """Compute the real and generated examples.
        Args:
            v (Tensor): The visible variable (orig).
        Returns:
            v (Tensor): The generagted variable (generated).
        """
        for _ in range(self.k):
            v, _ = self.gibbs_sample_vhv(v, cond)
        return v





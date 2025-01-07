#!/usr/bin/env python3
"""
VRPTR Model Definitions

Contains:
- RPTR: A mesh-based U-Net + Transformer + VAE-like regularization
- Supporting classes (Down, Up, ResPoolBlock, MeshConv, etc.)
"""

import os
import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse
from torch.nn.parameter import Parameter


class RPTR(nn.Module):
    """
    Resting-state to Task Prediction with Transformer and Regularization (VAE).
    """
    def __init__(self, mesh_dir, in_ch, out_ch, max_level=2, min_level=0, fdim=64):
        super().__init__()
        self.mesh_dir = mesh_dir
        self.fdim = fdim
        self.max_level = max_level
        self.min_level = min_level
        self.levels = max_level - min_level

        # U-Net style
        self.in_conv = MeshConv(in_ch, fdim, self._mesh_file(max_level), stride=1)
        self.out_conv = MeshConv(fdim, out_ch, self._mesh_file(max_level), stride=1)
        
        # Down path
        self.down = nn.ModuleList()
        for i in range(self.levels - 1):
            self.down.append(Down(fdim * (2**i),
                                  fdim * (2**(i+1)),
                                  max_level - i - 1,
                                  mesh_dir))
        self.down.append(Down(fdim * (2**(self.levels - 1)),
                              fdim * (2**(self.levels - 1)),
                              min_level,
                              mesh_dir))
        
        # Up path
        self.up = nn.ModuleList()
        for i in range(self.levels - 1):
            self.up.append(Up(fdim * (2**(self.levels - i)),
                              fdim * (2**(self.levels - i - 2)),
                              min_level + i + 1,
                              mesh_dir))
        self.up.append(Up(fdim * 2,
                          fdim,
                          max_level,
                          mesh_dir))
        
        # Transformer
        # Example shapes here are placeholders (dim_model=32, out_dim=32 or 2562).
        self.linear = nn.Linear(2562, 32)
        self.transformer_encoder = Transformer(dim_model=32, out_dim=32,
                                               num_heads=8, num_encoder_layers=6,
                                               dropout_p=0.1)
        self.transformer_decoder = Transformer(dim_model=32, out_dim=2562,
                                               num_heads=8, num_encoder_layers=6,
                                               dropout_p=0.1)
        
        # VAE Regularizer
        self.mu = nn.Linear(32, 32)
        self.sigma = nn.Linear(32, 32)
        self.norm = torch.distributions.Normal(0, 1)
        self.kl = 0.0  # track KL divergence

    def forward(self, x):
        """
        Forward pass:
        x: [batch, in_ch, #vertices]
        Returns: [batch, out_ch, #vertices]
        """
        # Encoder
        x_enc = [self.in_conv(x)]
        for i in range(self.levels):
            x_enc.append(self.down[i](x_enc[-1]))
        
        # Transformer (bottleneck)
        # x_enc[-1] assumed shape: [batch, channels, 2562], for example
        # Flatten channels => transform => reshape, etc., as needed
        z = self.linear(x_enc[-1])              # [batch, channels, 32]?
        z = self.transformer_encoder(z)         # [batch, seq_len, 32] => out
        # VAE reparam
        mu = self.mu(z)
        sigma = self.sigma(z)
        eps = self.norm.sample(mu.shape).to(mu.device)
        z = mu + torch.exp(0.5 * sigma) * eps
        
        # KL divergence approximation
        # Common VAE formula is .5 * sum(exp(sigma) + mu^2 - 1 - sigma). We'll keep your variant.
        self.kl = (torch.exp(mu) + sigma**2 - 1.0 - sigma).mean()
        
        # Task decoder
        z = self.transformer_decoder(z)
        # up path
        x_ = self.up[0](z, x_enc[-2])
        for i in range(self.levels - 1):
            x_ = self.up[i+1](x_, x_enc[-3 - i])
        out = self.out_conv(x_)
        return out

    def _mesh_file(self, level):
        return os.path.join(self.mesh_dir, f"icosphere_{level}.pkl")


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, level, mesh_dir):
        super().__init__()
        self.conv = ResPoolBlock(in_ch, in_ch, out_ch, level+1, coarsen=True, mesh_dir=mesh_dir)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, level, mesh_dir):
        super().__init__()
        mesh_file = os.path.join(mesh_dir, f"icosphere_{level}.pkl")
        half_in = in_ch // 2
        self.up = MeshConvTranspose(half_in, half_in, mesh_file,
                                    stride=2, level=level, mesh_dir=mesh_dir)
        self.conv = ResPoolBlock(in_ch, out_ch, out_ch, level, coarsen=False, mesh_dir=mesh_dir)

    def forward(self, x1, x2):
        x1_up = self.up(x1)
        x = torch.cat([x2, x1_up], dim=1)
        return self.conv(x)


class ResPoolBlock(nn.Module):
    """
    A block that can optionally downsample (coarsen) the mesh via MaxPool,
    and applies a residual connection with mesh-based convolution.
    """
    def __init__(self, in_chan, neck_chan, out_chan, level, coarsen, mesh_dir):
        super().__init__()
        self.coarsen = coarsen
        l = level - 1 if coarsen else level
        mesh_file = os.path.join(mesh_dir, f"icosphere_{l}.pkl")
        
        self.conv1 = nn.Conv1d(in_chan, neck_chan, kernel_size=1, stride=1)
        self.conv2 = MeshConv(neck_chan, neck_chan, mesh_file, stride=1)
        self.conv3 = nn.Conv1d(neck_chan, out_chan, kernel_size=1, stride=1)

        self.relu = nn.GELU()
        self.bn1 = nn.BatchNorm1d(neck_chan)
        self.bn2 = nn.BatchNorm1d(neck_chan)
        self.bn3 = nn.BatchNorm1d(out_chan)

        self.pool = MaxPool(mesh_dir, level)
        
        self.diff_chan = (in_chan != out_chan)
        if self.diff_chan or coarsen:
            self.conv_ = nn.Conv1d(in_chan, out_chan, kernel_size=1, stride=1)
            self.bn_ = nn.BatchNorm1d(out_chan)

            if coarsen:
                self.seq2 = nn.Sequential(self.conv_, self.pool, self.bn_)
            else:
                self.seq2 = nn.Sequential(self.conv_, self.bn_)
        else:
            self.seq2 = None

        if coarsen:
            self.seq1 = nn.Sequential(
                self.conv1, self.pool, self.bn1, self.relu,
                self.conv2, self.bn2, self.relu,
                self.conv3, self.bn3
            )
        else:
            self.seq1 = nn.Sequential(
                self.conv1, self.bn1, self.relu,
                self.conv2, self.bn2, self.relu,
                self.conv3, self.bn3
            )

    def forward(self, x):
        # Residual path
        if self.seq2 is not None:
            x2 = self.seq2(x)
        else:
            x2 = x
        
        x1 = self.seq1(x)
        out = x1 + x2
        return self.relu(out)


class MeshConv(nn.Module):
    """
    Standard mesh-based convolution as described in your references.
    """
    def __init__(self, in_channels, out_channels, mesh_file, stride=1, bias=True):
        super().__init__()
        assert stride in [1, 2]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncoeff = 4  # identity, laplacian, gradEW, gradNS
        self.coeffs = nn.Parameter(torch.Tensor(out_channels, in_channels, self.ncoeff))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._init_parameters()

        with open(mesh_file, "rb") as f:
            pkl_data = pickle.load(f)
        
        self.nv = pkl_data['V'].shape[0]
        self.G = sparse2tensor(pkl_data['G'])     # gradient
        self.NS = torch.tensor(pkl_data['NS'], dtype=torch.float32)
        self.EW = torch.tensor(pkl_data['EW'], dtype=torch.float32)

        if stride == 2:
            self.nv_prev = pkl_data['nv_prev']
            L = sparse2tensor(pkl_data['L'].tocsr()[:self.nv_prev].tocoo())
            F2V = sparse2tensor(pkl_data['F2V'].tocsr()[:self.nv_prev].tocoo())
        else:
            self.nv_prev = self.nv
            L = sparse2tensor(pkl_data['L'].tocoo())
            F2V = sparse2tensor(pkl_data['F2V'].tocoo())

        self.L = L
        self.F2V = F2V

    def _init_parameters(self):
        n = self.in_channels * self.ncoeff
        stdv = 1.0 / math.sqrt(n)
        self.coeffs.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        x shape: [batch, in_channels, #vertices]
        """
        grad_face = spmatmul(x, self.G)
        grad_face = grad_face.view(x.size(0), x.size(1), 3, -1).permute(0, 1, 3, 2)
        
        laplacian = spmatmul(x, self.L)
        identity = x[..., :self.nv_prev]

        grad_face_ew = torch.sum(grad_face * self.EW, dim=-1)
        grad_face_ns = torch.sum(grad_face * self.NS, dim=-1)

        grad_vert_ew = spmatmul(grad_face_ew, self.F2V)
        grad_vert_ns = spmatmul(grad_face_ns, self.F2V)

        feat = [identity, laplacian, grad_vert_ew, grad_vert_ns]
        out = torch.stack(feat, dim=-1)  # shape: [batch, in_ch, #vertices, 4]
        
        # Perform a matrix multiply-like operation with learned coeffs
        # out.unsqueeze(1) => [batch, 1, in_ch, #vertices, 4]
        # self.coeffs.unsqueeze(2) => [out_ch, in_ch, 1, 4]
        # Sum over in_ch & 4 to produce shape [batch, out_ch, #vertices]
        out = torch.sum(torch.sum(out.unsqueeze(1) * self.coeffs.unsqueeze(2), dim=2), dim=-1)
        
        if self.bias is not None:
            out += self.bias.unsqueeze(-1)
        return out


class MeshConvTranspose(nn.Module):
    """
    "Transpose" version of MeshConv for upsampling the mesh from coarser to finer resolution.
    """
    def __init__(self, in_channels, out_channels, mesh_file,
                 stride=2, bias=True, level=-1, mesh_dir=None):
        super().__init__()
        assert stride == 2, "MeshConvTranspose typically uses stride=2"
        assert level > 0, "Level must be > 0 for upsampling"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncoeff = 4
        
        self.coeffs = nn.Parameter(torch.Tensor(out_channels, in_channels, self.ncoeff))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._init_parameters()

        with open(mesh_file, "rb") as f:
            pkl_data = pickle.load(f)

        self.nv = pkl_data['V'].shape[0]
        self.G = sparse2tensor(pkl_data['G'])
        self.NS = torch.tensor(pkl_data['NS'], dtype=torch.float32)
        self.EW = torch.tensor(pkl_data['EW'], dtype=torch.float32)
        
        self.L = sparse2tensor(pkl_data['L'].tocoo())
        self.F2V = sparse2tensor(pkl_data['F2V'].tocoo())
        
        # Mapping from coarse level to current
        # e.g. "icosphere_{level-1}_to_icosphere_{level}_vertices.npy"
        if mesh_dir is None:
            raise ValueError("mesh_dir must be provided for MeshConvTranspose.")
        mapping_file = os.path.join(mesh_dir,
            f"icosphere_{level-1}_to_icosphere_{level}_vertices.npy")
        self.vertices_to_prev_lvl = np.load(mapping_file)

    def _init_parameters(self):
        n = self.in_channels * self.ncoeff
        stdv = 1.0 / math.sqrt(n)
        self.coeffs.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x_coarse):
        """
        x_coarse shape: [batch, in_channels, #coarse_vertices]
        We upsample to #finer_vertices by placing coarse data into the
        corresponding indices, then convolving similarly to MeshConv.
        """
        device = x_coarse.device
        batch_size, _, _ = x_coarse.size()

        # Initialize finer-level input
        x_fine = torch.zeros(batch_size, self.in_channels, self.nv, device=device)
        x_fine[..., self.vertices_to_prev_lvl] = x_coarse

        grad_face = spmatmul(x_fine, self.G)
        grad_face = grad_face.view(batch_size, self.in_channels, 3, -1).permute(0, 1, 3, 2)
        
        laplacian = spmatmul(x_fine, self.L)
        identity = x_fine
        
        grad_face_ew = torch.sum(grad_face * self.EW, dim=-1)
        grad_face_ns = torch.sum(grad_face * self.NS, dim=-1)
        
        grad_vert_ew = spmatmul(grad_face_ew, self.F2V)
        grad_vert_ns = spmatmul(grad_face_ns, self.F2V)
        
        feat = [identity, laplacian, grad_vert_ew, grad_vert_ns]
        out = torch.stack(feat, dim=-1)
        out = torch.sum(torch.sum(out.unsqueeze(1) * self.coeffs.unsqueeze(2), dim=2), dim=-1)
        
        if self.bias is not None:
            out += self.bias.unsqueeze(-1)
        return out


class MaxPool(nn.Module):
    """
    Max-pool a region of the mesh from one level to the next coarser level.
    """
    def __init__(self, mesh_dir, level):
        super().__init__()
        self.level = level
        if self.level > 0:
            vertices_file = os.path.join(mesh_dir,
                f"icosphere_{level-1}_to_icosphere_{level}_vertices.npy")
            self.vertices_to_prev_lvl = np.load(vertices_file)

            patch_file = os.path.join(mesh_dir,
                f"icosphere_{level}_neighbor_patches.npy")
            self.neighbor_patches = np.load(patch_file)

    def forward(self, x):
        if self.level == 0:
            return x
        # x shape: [batch, channels, #vertices_finer]
        # Map data to coarser
        tmp = x[..., self.vertices_to_prev_lvl]
        # shape: [batch, channels, #coarser_vertices, patch_size]
        out, _ = torch.max(tmp[:, :, self.neighbor_patches], dim=-1)
        return out


class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for Transformer-based models.
    """
    def __init__(self, dim_model, dropout_p, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() *
                             (-math.log(10000.0) / dim_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # shape: [max_len, 1, dim_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: [batch, seq_len, dim_model] (batch_first=True)
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, 0, :]
        return self.dropout(x)


class Transformer(nn.Module):
    """
    A wrapper around a TransformerEncoder for demonstration.
    """
    def __init__(self, dim_model, out_dim, num_heads, num_encoder_layers, dropout_p):
        super().__init__()
        self.dim_model = dim_model
        self.positional_encoder = PositionalEncoding(dim_model, dropout_p)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                                   nhead=num_heads,
                                                   activation='gelu',
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_encoder_layers)
        self.output = nn.Linear(dim_model, out_dim)

    def forward(self, x):
        """
        x shape: [batch, seq_len, dim_model]
        """
        x_pe = self.positional_encoder(x)
        x_trans = self.transformer(x_pe)
        out = self.output(x_trans)
        return out


def sparse2tensor(m):
    """ Convert a scipy.sparse.coo_matrix to a torch.sparse.FloatTensor. """
    assert isinstance(m, sparse.coo.coo_matrix)
    indices = torch.LongTensor([m.row, m.col])
    values = torch.FloatTensor(m.data)
    return torch.sparse.FloatTensor(indices, values, torch.Size(m.shape))


def spmatmul(dense, sp):
    """
    Sparse matrix multiplication for mesh operations.
    
    dense: [batch, in_chan, #vertices]
    sp:    [new_len, #vertices] (torch.sparse.FloatTensor)
    
    Returns shape: [batch, in_chan, new_len].
    """
    bsz, in_ch, nv = dense.size()
    new_len = sp.size(0)
    # reshape => [nv, in_ch*batch]
    dense_2d = dense.permute(2, 1, 0).contiguous().view(nv, -1)
    out_2d = torch.spmm(sp, dense_2d)  # => [new_len, in_ch*batch]
    out_3d = out_2d.view(new_len, in_ch, bsz).permute(2, 1, 0)
    return out_3d

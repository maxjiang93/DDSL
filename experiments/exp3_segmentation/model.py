import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torchvision.models as models
import sys; sys.path.append("../../ddsl")
from ddsl import DDSL_phys
from math import pi

EPS = 1e-7

class SmoothnessLoss(nn.Module):
    def __init__(self, degree=2):
        super(SmoothnessLoss, self).__init__()
        self.degree = degree

    def forward(self, V):
        """
        :param V: shape (N, L, 2)
        """
        L = V.shape[1]
        seq_next = list(range(1, L)) + [0]
        seq_prev = [L-1] + list(range(L-1)) 
        V_next = V[:, seq_next]
        V_prev = V[:, seq_prev]
        vec_n = V_next - V
        vec_p = V - V_prev
        cos_theta = torch.sum(vec_n * vec_p, dim=-1)/torch.norm(vec_n, dim=-1)/torch.norm(vec_p, dim=-1)
        cos_theta = torch.clamp(cos_theta, min=-1+EPS, max=1-EPS)
        theta = pi - torch.acos(cos_theta)
        loss = ((-1) ** self.degree) * (theta / pi - 1) ** self.degree
        return torch.mean(loss, dim=1)

class BatchedRasterLoss2D(nn.Module):
    def __init__(self, npoints=128, res=224, device=torch.device("cuda"), loss='l2', smoothing='gaussian', sig=1.5, return_raster=False):
        super(BatchedRasterLoss2D, self).__init__()
        assert(loss in ['l1', 'l2'])
        self.ddsl = DDSL_phys([res] * 2, t=[1] * 2, j=2, smoothing=smoothing, sig=sig)
        self.npoints = npoints
        self.res = res
        self.loss = loss
        self.device = device
        E, D = self._get_E_D()
        self.register_buffer("E", E)
        self.register_buffer("D", D)
        self.return_raster = return_raster

    def forward(self, V, f_target):
        """
        :param V: [N, npoints, 2]
        :param f_target: [N, res, res]
        """
        V = V.double()
        b = V.shape[0]

        f = torch.stack(tuple([self.ddsl(V[i], self.E, self.D).squeeze(-1) for i in range(b)]), dim=0) # [N, res, res]
        # loss
        if self.loss == 'l1':
            l = torch.mean(torch.abs(f-f_target), dim=(1,2))
        elif self.loss == 'l2':
            l = torch.mean((f-f_target)**2, dim=(1,2))
        if not self.return_raster:
            return l, None
        else:
            return l, f

    def _get_E_D(self):
        perm = list(range(1, self.npoints)) + [0]
        seq0 = torch.arange(self.npoints)
        seq1 = seq0[perm]
        E = torch.stack((seq0, seq1), dim=-1)
        D = torch.ones(E.shape[0], 1, dtype=torch.float64)
        return E, D

class PeriodicUpsample1D(nn.Module):
    def __init__(self, in_channel, out_channel):
        """
        :param in_channel: input planes
        :param out_channel: output planes
        """
        super(PeriodicUpsample1D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1d = nn.Conv1d(in_channel, out_channel, 3, 1, 0)

    def forward(self, x):
        """
        :param x: tensor of shape (N, C, L)
        :return tensor of shape (N, C, 2L)
        """
        # pad zeros
        x = torch.stack((torch.zeros_like(x), x), dim=-1).view(x.shape[0], x.shape[1], -1)
        x = torch.cat((x[..., -1:], x, x[..., :1]), dim=-1)
        x = self.conv1d(x)
        return x

class PolygonNet(nn.Module):
    def __init__(self, encoder=models.resnet50(pretrained=True), npoints=128):
        super(PolygonNet, self).__init__()
        self.npoints = npoints
        self.encoder = nn.Sequential(*list(encoder.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU()
            )
        # output of encoder (after projection) is  [N, 512, 1]
        self.decoder = nn.Sequential(
            PeriodicUpsample1D(512, 256), # [N, 256,   2]
            nn.BatchNorm1d(256),
            nn.ReLU(),
            PeriodicUpsample1D(256, 128), # [N, 128,   4]
            nn.BatchNorm1d(128),
            nn.ReLU(),
            PeriodicUpsample1D(128,  64), # [N,  64,   8]
            nn.BatchNorm1d(64),
            nn.ReLU(),
            PeriodicUpsample1D( 64,  32), # [N,  32,  16]
            nn.BatchNorm1d(32),
            nn.ReLU(),
            PeriodicUpsample1D( 32,  16), # [N,  16,  32]
            nn.BatchNorm1d(16),
            nn.ReLU(),
            PeriodicUpsample1D( 16,   8), # [N,   8,  64]
            nn.BatchNorm1d(8),
            nn.ReLU(),
            PeriodicUpsample1D(  8,   4), # [N,   4, 128]
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Conv1d(4, 2, 1, 1, 0),     # [N,   2, 128]
            nn.Sigmoid() # squash output to (0,1)
            )
        offsets = 1e-4*torch.rand(self.npoints, 2, dtype=torch.float64)
        self.register_buffer("offsets", offsets)

    def forward(self, x):
        """
        :param x: [N, C, H, W], where C=3, H=W=224, input image
        :return [N, 128, 2] segmentation polygons
        """
        x = self.encoder(x)
        x = x.squeeze(-1).squeeze(-1) # [N, 2048]
        x = self.projection(x).unsqueeze(-1) # [N, 512, 1]
        x = self.decoder(x)
        x = x.permute(0, 2, 1).double()
        x += self.offsets
        return x


# NEW ARCHITECTURE

class DecodeLeaf(nn.Module):
    def __init__(self, in_channel):
        """
        :param in_channel: input planes
        """
        super(DecodeLeaf, self).__init__()
        self.in_channel = in_channel
        self.conv1x1 = nn.Conv1d(in_channel, 1, 1, 1, 0)
        self.scale = Parameter(0.5*torch.ones(1))

    def forward(self, x):
        """
        :param x: tensor of shape (N, C, L)
        :return tensor of shape (N, L)
        """
        x = self.conv1x1(x).squeeze(1) # (N, L)
        x = self.scale * 0.5 * torch.tanh(x)
        return x

class DropoutLayer(nn.Module):
    def __init__(self, p=0.5):
        super(DropoutLayer, self).__init__()
        self.p = p
        self.drop = nn.Dropout2d(p=p)

    def forward(self, x):
        """
        :input x: shape (N, C, L)
        """
        x = x.unsqueeze(-1) # (N, C, L, 1)
        x = self.drop(x)
        x = x.squeeze(-1)
        return x

class PolygonNet2(nn.Module):
    def __init__(self, encoder=models.resnet50(pretrained=True), nlevels=5, dropout=False, feat=256):
        super(PolygonNet2, self).__init__()
        self.p = 0.5 if dropout else 0
        self.base = feat
        self.nlevels = nlevels
        self.npoints = 3+3*(2**nlevels-1)
        self.encoder = nn.Sequential(*list(encoder.children())[:-1])
        self.projection = nn.Sequential(
            nn.Dropout(p=self.p),
            nn.Linear(2048, 3*self.base),
            nn.ReLU(),
            )
        # output of encoder (after projection and reshape) is  [N, 256, 3]
        self.decoder_stem = []
        for i in range(self.nlevels-1):
            c_in = int(self.base / (2**i))
            c_out = int(self.base / (2**(i+1)))
            self.decoder_stem.append(
                nn.Sequential(
                    PeriodicUpsample1D(c_in, c_out),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU()
                    )
                )

        self.decoder_leaf = []
        for i in range(self.nlevels):
            c_in = int(self.base / (2**i))
            self.decoder_leaf.append(DecodeLeaf(c_in))

        self.decoder_stem = nn.ModuleList(self.decoder_stem)
        self.decoder_leaf = nn.ModuleList(self.decoder_leaf)

        self.conv1x1 = nn.Conv1d(self.base, 2, 1, 1, 0)
        
        offsets = 1e-4*torch.rand(self.npoints, 2, dtype=torch.float64)
        rot = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
        self.register_buffer("offsets", offsets)
        self.register_buffer("rot", rot)

    def forward(self, x):
        """
        :param x: [N, C, H, W], where C=3, H=W=224, input image
        :return [N, 128, 2] segmentation polygons
        """
        x = self.encoder(x)
        x = x.squeeze(-1).squeeze(-1) # [N, 2048]
        x = self.projection(x).view(-1, self.base, 3) # [N, 256, 3]
        base_triangle = torch.sigmoid(self.conv1x1(x).permute(0, 2, 1)) # [N, 3, 2]

        stem_values = [x]
        leaf_values = []
        for i in range(self.nlevels-1):
            stem_values.append(self.decoder_stem[i](stem_values[-1]))
        for i in range(self.nlevels):    
            leaf_values.append(self.decoder_leaf[i](stem_values[i])) # [N, 3*2**i]
        V = self._construct_polygon(base_triangle, leaf_values)
        V += self.offsets
        return V

    def _construct_polygon(self, base_triangle, leaf_values):
        V = base_triangle
        for diff in leaf_values:
            V = self._add_resolution(V, diff)
        V = V.double()
        return V

    def _add_resolution(self, V, diff):
        """
        :param V: shape [N, nv, 2]
        :param diff: shape [N, nv]
        """
        N = V.shape[0]
        nv = V.shape[1]
        seq = list(range(1, nv)) + [0]
        V_next = V[:, seq, :] # [N, nv, 2]
        V_mid = (V + V_next) / 2
        V_dir = (V_next - V) / torch.norm(V_next - V, dim=2, keepdim=True) # [N, nv, 2]
        V_per = torch.matmul(V_dir, self.rot) # [N, nv, 2]
        V_new = V_mid + V_per * diff.unsqueeze(-1)
        
        V_out = torch.stack((V, V_new), dim=2).view(N, nv*2, 2)
        # V_out = torch.clamp(V_out, 0, 0.99) # clip output value
        return V_out

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

class PolygonNet3(nn.Module):
    def __init__(self, encoder=models.resnet50(pretrained=True), nlevels=5, dropout=False, feat=256):
        super(PolygonNet3, self).__init__()
        self.p = 0.5 if dropout else 0
        self.base = feat
        self.nlevels = nlevels
        self.npoints = 3+3*(2**nlevels-1)
        self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.projection = nn.Sequential(
            nn.Conv2d(2048, 16, 1, 1, 0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=self.p),
            Flatten(),
            nn.Linear(16*7*7, 3*self.base),
            nn.ReLU(),
            )
        # output of encoder (after projection and reshape) is  [N, 256, 3]
        self.decoder_stem = []
        for i in range(self.nlevels-1):
            c_in = int(self.base / (2**i))
            c_out = int(self.base / (2**(i+1)))
            self.decoder_stem.append(
                nn.Sequential(
                    PeriodicUpsample1D(c_in, c_out),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU()
                    )
                )

        self.decoder_leaf = []
        for i in range(self.nlevels):
            c_in = int(self.base / (2**i))
            self.decoder_leaf.append(DecodeLeaf(c_in))

        self.decoder_stem = nn.ModuleList(self.decoder_stem)
        self.decoder_leaf = nn.ModuleList(self.decoder_leaf)

        self.conv1x1 = nn.Conv1d(self.base, 2, 1, 1, 0)
        
        offsets = 1e-4*torch.rand(self.npoints, 2, dtype=torch.float64)
        rot = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
        self.register_buffer("offsets", offsets)
        self.register_buffer("rot", rot)

    def forward(self, x):
        """
        :param x: [N, C, H, W], where C=3, H=W=224, input image
        :return [N, 128, 2] segmentation polygons
        """
        x = self.encoder(x)
        x = x.squeeze(-1).squeeze(-1) # [N, 2048]
        x = self.projection(x).view(-1, self.base, 3) # [N, 256, 3]
        base_triangle = torch.sigmoid(self.conv1x1(x).permute(0, 2, 1)) # [N, 3, 2]

        stem_values = [x]
        leaf_values = []
        for i in range(self.nlevels-1):
            stem_values.append(self.decoder_stem[i](stem_values[-1]))
        for i in range(self.nlevels):    
            leaf_values.append(self.decoder_leaf[i](stem_values[i])) # [N, 3*2**i]
        V = self._construct_polygon(base_triangle, leaf_values)
        V += self.offsets
        return V

    def _construct_polygon(self, base_triangle, leaf_values):
        V = base_triangle
        for diff in leaf_values:
            V = self._add_resolution(V, diff)
        V = V.double()
        return V

    def _add_resolution(self, V, diff):
        """
        :param V: shape [N, nv, 2]
        :param diff: shape [N, nv]
        """
        N = V.shape[0]
        nv = V.shape[1]
        seq = list(range(1, nv)) + [0]
        V_next = V[:, seq, :] # [N, nv, 2]
        V_mid = (V + V_next) / 2
        V_dir = (V_next - V) / torch.norm(V_next - V, dim=2, keepdim=True) # [N, nv, 2]
        V_per = torch.matmul(V_dir, self.rot) # [N, nv, 2]
        V_new = V_mid + V_per * diff.unsqueeze(-1)
        
        V_out = torch.stack((V, V_new), dim=2).view(N, nv*2, 2)
        # V_out = torch.clamp(V_out, 0, 0.99) # clip output value
        return V_out

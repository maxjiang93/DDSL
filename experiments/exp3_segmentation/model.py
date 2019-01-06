import torch
import torch.nn as nn
import torchvision.models as models
import sys; sys.path.append("../../ddsl")
from ddsl import DDSL_phys

class BatchedRasterLoss2D(nn.Module):
	def __init__(self, npoints=128, res=224, loss='l2', smooth='gaussian', sig=1.5):
		super(RasterLoss2D, self).__init__()
		assert(loss in ['l1', 'l2'])
		self.ddsl = DDSL_phys([res] * 2, t=[1] * 2, j=2, smooth=smooth, sig=sig)
		self.npoints = npoints
		self.res = res
		self.loss = loss
		self.E, self.D = self._get_E()

	def forward(self, V, f_target):
		"""
		:param V: [N, npoints, 2]
		:param f_target: [N, res, res]
		"""
		b = V.shape[0]
		f = torch.stack(tuple([self.ddsl(V[i], self.E, self.D) for i in range(b)]), dim=0) # [N, res, res]
		# loss
		if self.loss == 'l1':
			l = torch.mean(torch.abs(f-f_target), dim=(1,2))
		elif self.loss == 'l2':
			l = torch.mean((f-f_target)**2, dim=(1,2))
		return l

	def _get_E_D(self):
		perm = list(range(1, self.npoints)) + [0]
		seq0 = torch.arange(self.npoints)
		seq1 = seq[0][perm]
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
        # TODO: check output of encoder is  [N, 512,   1]
        self.decoder = nn.Sequential(
        	PeriodicUpsample1D(512, 256), # [N, 256,   2]
        	nn.BatchNorm1d(2),
        	nn.ReLU(),
        	PeriodicUpsample1D(256, 128), # [N, 128,   4]
        	nn.BatchNorm1d(4),
        	nn.ReLU(),
        	PeriodicUpsample1D(128,  64), # [N,  64,   8]
        	nn.BatchNorm1d(8),
        	nn.ReLU(),
        	PeriodicUpsample1D( 64,  32), # [N,  32,  16]
        	nn.BatchNorm1d(16),
        	nn.ReLU(),
        	PeriodicUpsample1D( 32,  16), # [N,  16,  32]
        	nn.BatchNorm1d(32),
        	nn.ReLU(),
        	PeriodicUpsample1D( 16,   8), # [N,   8,  64]
        	nn.BatchNorm1d(64),
        	nn.ReLU(),
        	PeriodicUpsample1D(  8,   4), # [N,   4, 128]
        	nn.BatchNorm1d(128),
        	nn.ReLU(),
        	nn.Conv1d(4, 2, 1, 1, 0)      # [N,   2, 128]
        	nn.Sigmoid() # squash output to (0,1)
        	)

    def forward(self, x):
    	"""
		:param x: [N, C, H, W], where C=3, H=W=224, input image
		:return [N, 128, 2] segmentation polygons
    	"""
    	x = self.encoder(x)
    	x = self.decoder(x)
    	x = x.permute(0, 2, 1)
    	return x

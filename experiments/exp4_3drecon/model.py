import os
import pickle
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torchvision.models as models
from meshcnn import MeshConv, MeshConv_transpose, ResBlock, spmatmul
import sys; sys.path.append("../../ddsl")
from ddsl import DDSL_phys


class SphereNet(nn.Module):
    def __init__(self, mesh_folder, nlevels=3, feat=256, encoder=models.resnet18(pretrained=True), deform=True):
        super(SphereNet, self).__init__()
        self.mesh_folder = mesh_folder
        self.encoder = nn.Sequential(*list(encoder.children())[:-1])
        self.nlevels = nlevels
        self.feat = feat
        self.deform = deform
        self.fc = nn.Conv2d(512, feat*12, kernel_size=1, stride=1)
        self.convs = [Up(in_ch=int(feat/(2**i)),
                         out_ch=int(feat/(2**(i+1))),
                         level=i+1,
                         mesh_folder=mesh_folder)
                      for i in range(nlevels)]
        self.convs = nn.ModuleList(self.convs)
        self.out = nn.Conv1d(int(feat/(2**nlevels)), 3, kernel_size=1, 
            stride=1)
        self.scale = Parameter(torch.tensor([0.1]))
        pkl = pickle.load(open(self._meshfile(nlevels), "rb"))
        v = torch.tensor(pkl['V']).unsqueeze(0).type(torch.float32)
        v = v / 4 + 0.5  # rescale and recenter base
        f = torch.tensor(pkl['F']).type(torch.int32)
        L = sparse2tensor(pkl['L'].tocoo())
        self.register_buffer('L', L)
        self.register_buffer('v', v)
        self.register_buffer('f', f)

    def forward(self, x):
        # feed image through resnet
        x = self.encoder(x)
        # print(x.shape)

        # linear
        x = self.fc(x)
        # print(x.shape)

        # reshape
        b = x.shape[0]
        x = x.view(b, self.feat, 12)  # 12 is number of vertices at ico0
        
        # mesh conv
        for conv in self.convs:
            x = conv(x)

        # out conv
        x = self.out(x)

        if self.deform:
            x = self.scale * torch.tanh(x)  # learnable scale
            x = x.permute(0, 2, 1)
            x += self.v
        else:
            x = x.permute(0, 2, 1)
            x = torch.sigmoid(x)
        return x

    def _meshfile(self, i):
        return os.path.join(self.mesh_folder, "icosphere_{}.pkl".format(i))

    @property
    def face(self):
        return self.f
    

class SphereNet2(nn.Module):
    def __init__(self, mesh_folder, nlevels=3, feat=256, encoder=models.resnet18(pretrained=True)):
        super(SphereNet, self).__init__()
        self.mesh_folder = mesh_folder
        self.encoder = nn.Sequential(*list(encoder.children())[:-1])
        self.nlevels = nlevels
        self.feat = feat
        self.fc = nn.Conv2d(512, feat*12, kernel_size=1, stride=1)
        self.convs = [Up(in_ch=int(feat/(2**i)),
                         out_ch=int(feat/(2**(i+1))),
                         level=i+1,
                         mesh_folder=mesh_folder)
                      for i in range(nlevels)]
        self.conv_out = [nn.Conv1d(in_channels=int(feat/(2**i)),
                                   out_channels=3, kernel_size=1, stride=1) 
                                   for i in range(nlevels+1)]
        self.convs = nn.ModuleList(self.convs)
        self.conv_out = nn.ModuleList(self.conv_out)

        self.scale = Parameter(torch.tensor([0.1]*nlevels))
        pkl = pickle.load(open(self._meshfile(nlevels), "rb"))
        v = torch.tensor(pkl['V']).unsqueeze(0).type(torch.float32)
        v = v / 4 + 0.5  # rescale and recenter base
        f = torch.tensor(pkl['F']).type(torch.int32)
        self.register_buffer('v', v)
        self.register_buffer('f', f)

    def forward(self, x):
        # feed image through resnet
        x = self.encoder(x)

        # linear
        x = self.fc(x)

        # reshape
        b = x.shape[0]
        x = x.view(b, self.feat, 12)  # 12 is number of vertices at ico0
        xout = torch.sigmoid(self.conv_out[0](x))
        # mesh conv
        for i, (conv, convout) in enumerate(zip(self.convs, self.conv_out[1:])):
            nv_prev = x.shape[-1]
            x = conv(x)
            x_ = convout(x)
            xout = torch.matmul(xout, conv.intp.permute(1,0))
            xout[..., nv_prev:] += self.scale[i] * torch.tanh(x_[..., nv_prev])

        return xout

    def _meshfile(self, i):
        return os.path.join(self.mesh_folder, "icosphere_{}.pkl".format(i))

    @property
    def face(self):
        return self.f


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, level, mesh_folder, bias=True):
        """
        use mesh_file for the mesh of one-level up
        """
        super().__init__()
        mesh_file = os.path.join(mesh_folder, "icosphere_{}.pkl".format(level))
        half_out = int(out_ch/2)
        self.up = MeshConv_transpose(in_ch, in_ch, mesh_file, stride=2)
        self.conv = ResBlock(in_ch, half_out, out_ch, level, False, mesh_folder)
        self.register_buffer('intp', self.conv.intp)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class BatchedRasterLoss3D(nn.Module):
    def __init__(self, E, res=32, loss='l1', smoothing='gaussian', sig=1.5, return_raster=False):
        super(BatchedRasterLoss3D, self).__init__()
        assert(loss in ['l1', 'l2'])
        self.ddsl = DDSL_phys([res]*3, t=[1]*3, j=3, smoothing=smoothing, sig=sig, elem_batch=640)
        self.res = res
        self.loss = loss
        D = torch.ones(E.shape[0], 1).double()
        self.register_buffer("E", E)
        self.register_buffer("D", D)
        self.return_raster = return_raster

    def forward(self, V, f_target):
        """
        :param V: [N, npoints, 3]
        :param f_target: [N, res, res, res]
        """
        V = V.double()
        V += 1e-4 * (torch.rand_like(V)-0.5)
        b = V.shape[0]
        f = torch.stack(tuple([self.ddsl(V[i], self.E, self.D).squeeze(-1) for i in range(b)]), dim=0) # [N, res, res, res]
        f = f.float()
        # loss
        if self.loss == 'l1':
            l = torch.mean(torch.abs(f-f_target))
        elif self.loss == 'l2':
            l = torch.mean((f-f_target)**2)
        if not self.return_raster:
            return l, None
        else:
            return l, f


class ChamferLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ChamferLoss, self).__init__()
        assert(reduction in ['mean', 'sum', 'none'])
        if reduction == 'mean':
            self.reduce = lambda x: torch.mean(x) 
        elif reduction == 'sum':
            self.reduce = lambda x: torch.avg(x)
        else:
            self.reduce = lambda x: x

    def forward(self, trgt, pred):
        """
        Args:
          trgt: [b, n, 3] points for target
          pred: [b, m, 3] points for predictions
        Returns:
          accuracy, complete, chamfer
        """
        trgt = trgt.unsqueeze(2)
        pred = pred.unsqueeze(1)
        diff = trgt - pred  # [b, n, m, 3]
        dist = torch.norm(diff, dim=-1)  # [b, n, m]
        complete = torch.mean(dist.min(2)[0], dim=1)  # [b]
        accuracy = torch.mean(dist.min(1)[0], dim=1)  # [b]
        
        complete = self.reduce(complete)
        accuracy = self.reduce(accuracy)
        chamfer = 0.5 * (complete + accuracy)
        return accuracy, complete, chamfer

class MeshSampler(nn.Module):
    def __init__(self, F, nsamp=1000):
        """
        Args:
          F: [#f, 3], face connectivity matrix.
        """
        super(MeshSampler, self).__init__()
        self.nsamp = nsamp
        self.register_buffer('F', F)

    def forward(self, V):
        """
        Args:
          V: [N, #v, 3], batched vertex matrix. shared connectivity.
        """
        F = self.F
        device = V.device
        b = V.shape[0]
        V0 = V[:, F[:, 0]]
        V01 = V[:, F[:, 1]] - V0  # [N, #f, 3]
        V02 = V[:, F[:, 2]] - V0  # [N, #f, 3]
        face_area = 0.5 * torch.norm(V01.cross(V02, dim=-1), dim=-1)  # [N, #f]
        tot_area = torch.sum(face_area, dim=-1, keepdim=True)  # [N, 1]
        face_prob = face_area / tot_area  # [N, #f]
        face_index = torch.multinomial(face_prob, num_samples=self.nsamp, 
                                       replacement=True).to(device)  # [N, #s]
        samp_index = torch.arange(b, dtype=torch.long).to(device).unsqueeze(-1).repeat(1, self.nsamp)
        samp_vecs = torch.stack((V01, V02), dim=-1)[samp_index, face_index]  # [N, #s, 3, 2]
        samp_orig = V0[samp_index, face_index]

        rand_scale = torch.rand(b, self.nsamp, 2).to(device)  # [N, #s, 2]
        rand_flipind = torch.sum(rand_scale, dim=-1, keepdim=False) > 1  # [N, #s]
        rand_scale[rand_flipind] -= 1
        rand_scale = torch.abs(rand_scale)
        rand_scale = rand_scale.unsqueeze(2)  # [N, #s, 1, 2]
        samp_pts = samp_orig + torch.sum(samp_vecs * rand_scale, dim=-1)  # [N, #s, 3]
        return samp_pts


class LaplacianLoss(nn.Module):
    def __init__(self, mesh_file):
        pass


if __name__ == '__main__':
    pass
    # net = SphereNet("mesh_files")
    # input = torch.rand([2,3,224,224])
    # out = net(input)
    # print(out.shape)

    # import time
    # a = torch.rand(12, 2048, 3).cuda()
    # b = torch.rand(12, 2048, 3).cuda()
    # cl = ChamferLoss('mean', 'chamfer')
    # t0=time.time(); cl(a,b); print(time.time()-t0)

    # def saveobj(filename, verts, faces):
    #     if faces is not None and faces.min() == 0:
    #         faces += 1
    #     thefile = open(filename, 'w')
    #     for item in verts:
    #         thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))

    #     if faces is not None:
    #         for item in faces:
    #           thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0],item[1],item[2]))  

    #     thefile.close()

    # import trimesh
    # p = pickle.load(open("mesh_files/icosphere_0.pkl", "rb"))
    # v, f = torch.tensor(p['V']).type(torch.float32), torch.tensor(p['F']).type(torch.long)
    # ms = MeshSampler(f, nsamp=10000)
    # vs = ms(v.unsqueeze(0)).squeeze(0).numpy()
    # saveobj("samp_pts.obj", vs, None)

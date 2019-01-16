from glob import glob
import os
import skimage.transform as transform
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import argparse
from model import PolygonNet2
from tqdm import tqdm
import json
import numpy as np
import PIL
from random import shuffle
import cv2

EPS = 1e-7

def batch_smooth(polygons, steps=0, lambda_=.5):
    """
    batched laplacian smoothing for post-processing
    :param polygons: shape (N, L, 2)
    :param steps: number of steps
    """
    L = polygons.shape[1]
    for _ in range(steps):
        next_ = list(range(1, L)) + [0]
        prev_ = [L-1] + list(range(L-1))
        lap = 0.5*(polygons[:, next_]+polygons[:, prev_]) - polygons
        polygons += lambda_ * lap
    return polygons


def batch_transform(inputs, transform):
    """
    :param inputs: shape (N, H, W, 3)
    :param transform: transform function for a single image
    :return (N, 3, H, W) tensor
    """
    if np.issubdtype(inputs.dtype, np.floating) and inputs.max() <= 1:
        inputs = (inputs * 255).astype(np.uint8)

    tensors = []
    for i in range(inputs.shape[0]):
        img = PIL.Image.fromarray(inputs[i])
        img_tensor = transform(img) # [3, H, W]
        tensors.append(img_tensor)

    return torch.stack(tensors, dim=0)


class visualizer(object):
    def __init__(self, model, transform, mapping_dir, img_dir, partition='test', skip_multicomponents=False, export_gt=True, smooth_steps=3, lambda_=.5, rand_samp=False):
        assert(partition in ['train', 'test', 'val'])
        # partition names named differently
        if partition == 'test':
            self.p = 'val'
        elif partition == 'val':
            self.p = 'train_val'
        elif partition == 'train':
            self.p = 'train'
        self.class_filter = [
                "car",
                "truck",
                "train",
                "bus",
                "motorcycle",
                "bicycle",
                "rider",
                "person"
            ]
        self.model = model
        self.transform = transform
        self.mapping_dir = mapping_dir
        self.img_dir = img_dir
        self.skip_multicomponents = skip_multicomponents
        self.export_gt = export_gt
        self.flist = sorted(glob(os.path.join(mapping_dir, self.p, "*", "*.json")))
        self.rand_samp = rand_samp
        if self.rand_samp:
            shuffle(self.flist)
        self.min_poly_len = 3
        self.max_poly_len = 71
        self.min_area = 100
        self.sub_th = 0
        self.smooth_steps = smooth_steps
        self.lambda_ = lambda_

    def __len__(self):
        return len(self.flist)
        
    def __getitem__(self, file):
        """
        Get item by id or png file name of image 
        """
        if isinstance(file, int):
            mfile = self.flist[file]
        elif isinstance(file, str):
            mfile = file.split('/')[-1].split('.')[0].replace('_leftImg8bit', '')
            city = mfile.split('_')[0]
            mfile = os.path.join(self.mapping_dir, self.p, city, mfile+".json")
        else:
            print("[!] ERROR: index by integer or by file name only!")
            assert(0)
            
        instances, _ = self.process_info(mfile)
        
        # read full image
        img_path = instances[0]['img_path'].split('/leftImg8bit/')[-1]
        img_path = os.path.join(self.img_dir, img_path)
        full_img = plt.imread(img_path)
        
        # prepare crops for input to neural network
        instance_dicts = [self.prepare_instance(ins, full_img) for ins in instances]
        crops = np.stack([ins['img'] for ins in instance_dicts], axis=0)
        
        # feed to neural net
        self.model.eval()
        inputs = batch_transform(crops, self.transform)
        pred_poly = self.model(inputs).detach().cpu().numpy()
        pred_poly = pred_poly[..., ::-1]

        # smooth polygons
        pred_poly = batch_smooth(pred_poly, steps=self.smooth_steps, lambda_=self.lambda_) 
        
        # backproject polygons onto image
        gt_polygons = [d['poly'] for d in instance_dicts]
        pd_polygons = [pred_poly[i] for i in range(pred_poly.shape[0])]
        colors = np.random.uniform(0, 255, size=(len(instance_dicts), 3))  

        PD_img = self.project_polygons(full_img, instance_dicts, pd_polygons, colors)
        if self.export_gt:
            GT_img = self.project_polygons(full_img, instance_dicts, gt_polygons, colors)
            return PD_img, GT_img, img_path
        else:
            return PD_img, img_path
    
    def project_polygons(self, img, instance_dicts, polygons, colors=None):
        assert(len(instance_dicts) == len(polygons))

        # colors
        contour_color = np.array((255., 133., 0.))
        if colors is None:
            colors = np.random.uniform(0, 255, size=(len(polygons), 3))         
        else:
            assert(colors.shape[0] == len(polygons))

        widths = [d['patch_w'] for d in instance_dicts]
        sps = [d['starting_point'] for d in instance_dicts]
        
        for i, (p, w, sp) in enumerate(zip(polygons, widths, sps)):
            poly = (p*w).astype(np.int)
            poly[:, 0] += sp[0]
            poly[:, 1] += sp[1]
            polygons[i] = poly

        pred_mask = np.zeros(list(img.shape), dtype=np.uint8) 

        for i, poly in enumerate(polygons):
            cv2.fillPoly(pred_mask, [poly], colors[i])
            cv2.polylines(pred_mask, [poly], True, contour_color, 4)

        pred_mask = np.clip(pred_mask, 0, 255)/255
        masked_img = np.clip(img+pred_mask*0.5, 0, 1)
        return masked_img
        
    def prepare_instance(self, instance, img):
        """
        Prepare a single instance
        """
        component = instance['components'][0]
        context_expansion = 0.15
    
        poly = np.array(component['poly'])

        xs = poly[:,0]
        ys = poly[:,1]

        bbox = instance['bbox']
        x0, y0, w, h = bbox

        x_center = x0 + (1+w)/2.
        y_center = y0 + (1+h)/2.

        widescreen = True if w > h else False
        
        if not widescreen:
            img = img.transpose((1, 0, 2))
            x_center, y_center, w, h = y_center, x_center, h, w
            xs, ys = ys, xs

        x_min = int(np.floor(x_center - w*(1 + context_expansion)/2.))
        x_max = int(np.ceil(x_center + w*(1 + context_expansion)/2.))
        
        x_min = max(0, x_min)
        x_max = min(img.shape[1] - 1, x_max)

        patch_w = x_max - x_min
        # NOTE: Different from before

        y_min = int(np.floor(y_center - patch_w / 2.))
        y_max = y_min + patch_w

        top_margin = max(0, y_min) - y_min

        y_min = max(0, y_min)
        y_max = min(img.shape[0] - 1, y_max)

        scale_factor = 224.0/patch_w

        patch_img = img[y_min:y_max, x_min:x_max, :]

        new_img = np.zeros([patch_w, patch_w, 3], dtype=np.float32)
        new_img[top_margin: top_margin + patch_img.shape[0], :, ] = patch_img

        new_img = transform.rescale(new_img, scale_factor, order=1, 
            preserve_range=True, multichannel=True)
        new_img = new_img.astype(np.float32)
        #assert new_img.shape == [self.opts['img_side'], self.opts['img_side'], 3]

        starting_point = [x_min, y_min-top_margin]

        xs = (xs - x_min) / float(patch_w)
        ys = (ys - (y_min-top_margin)) / float(patch_w)

        xs = np.clip(xs, 0 + EPS, 1 - EPS)
        ys = np.clip(ys, 0 + EPS, 1 - EPS)

        if not widescreen:
            # Now that everything is in a square
            # bring things back to original mode
            new_img = new_img.transpose((1,0,2))
            starting_point = [y_min-top_margin, x_min]
            xs, ys = ys, xs

        return_dict = {
            'img': new_img,
            'patch_w': patch_w,
            'top_margin': top_margin,
            'patch_shape': patch_img.shape,
            'scale_factor': scale_factor,
            'starting_point': starting_point,
            'widescreen': widescreen,
            'poly': np.array([xs, ys]).T
        }

        return return_dict
    
    def process_info(self, mfile):
        """
        Process a single json file
        """

        with open(mfile, 'r') as f:
            ann = json.load(f)

        examples = []
        skipped_instances = 0

        for instance in ann:
            components = instance['components']

            if instance['label'] not in self.class_filter:
                continue

            candidates = [c for c in components if len(c['poly']) >= self.min_poly_len]
            total_area = np.sum([c['area'] for c in candidates])
            candidates = [c for c in candidates if c['area'] > self.sub_th*total_area]
            candidates = [c for c in candidates if c['area'] >= self.min_area]

            if self.skip_multicomponents and len(candidates) > 1:
                skipped_instances += 1
                continue

            instance['components'] = candidates
            if candidates:
                examples.append(instance)

        return examples, skipped_instances 

    def filename(self, idx):
        return self.flist(idx)


def main():
    # Test settings
    parser = argparse.ArgumentParser(description='Segmentation')
    # parser.add_argument('--ckpt', type=str, default='/home/maxjiang/Codes/dsnet/experiments/exp3_segmentation/logs/net2_new2_resume_2019_01_10_16_01_56/checkpoint_polygonnet_best.pth.tar', help="path to checkpoint to load")
    parser.add_argument('--ckpt', type=str, default='/home/maxjiang/Codes/dsnet/experiments/exp3_segmentation/logs/net2_drop_l5_f256_2019_01_12_12_13_54/checkpoint_polygonnet_best.pth.tar', help="path to checkpoint to load")
    parser.add_argument('--nlevels', type=int, default=5, help="number of polygon levels, higher->finer")
    parser.add_argument('--feat', type=int, default=256, help="number of base feature layers")
    parser.add_argument('--dropout', action='store_true', help="dropout during training")
    parser.add_argument('--raw_img_dir', type=str, default='leftImg8bit', help='path to raw image directory')
    parser.add_argument('--mapping_dir', type=str, default='cityscapes_mapping', help='path to mapping directory')
    parser.add_argument('--gpuid', type=int, default=0, help='gpu id to use for evaluation')
    parser.add_argument('--output_dir', type=str, default='output_vis_full', help='directory to output images')
    parser.add_argument('--nsamples', type=int, default=10, help='number of samples to produce. 0 for all.')
    parser.add_argument('--export_gt', action='store_true', help='export ground truth images also')
    parser.add_argument('--skip_multicomponents', action='store_true', help='skip multiple components')
    parser.add_argument('--smooth_steps', type=int, default=3, help='number of laplacian smoothing steps')
    parser.add_argument('--lambda_', type=int, default=0.5, help='smoothing lambda in (0, 1)')
    parser.add_argument('--rand_samp', action='store_true', help='randomly draw samples to save')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    device = torch.device("cuda")

    # create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # get training / valid sets
    args.img_mean = [0.485, 0.456, 0.406]
    args.img_std = [0.229, 0.224, 0.225]
    normalize = torchvision.transforms.Normalize(mean=args.img_mean,
                                                  std=args.img_std)

    # DO NOT include horizontal and vertical flips in the composed transforms!
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(hue=.1, saturation=.1),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

    # load model
    model = PolygonNet2(nlevels=args.nlevels, dropout=args.dropout, feat=args.feat)
    model = nn.DataParallel(model)
    model.to(device)
    
    if os.path.isfile(args.ckpt):
        print("=> loading checkpoint '{}'".format(args.ckpt))
        checkpoint = torch.load(args.ckpt)
        args.best_miou = checkpoint['best_miou']
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            # simple hack for loading the old model
            sdict = checkpoint['state_dict']
            sdict['module.projection.1.weight'] = sdict.pop('module.projection.0.weight')
            sdict['module.projection.1.bias']   = sdict.pop('module.projection.0.bias')
            model.load_state_dict(sdict)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.ckpt, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.ckpt))

    # create and save visualizations
    vis = visualizer(model=model, transform=transform, mapping_dir=args.mapping_dir, img_dir=args.raw_img_dir, skip_multicomponents=args.skip_multicomponents, export_gt=args.export_gt, smooth_steps=args.smooth_steps, lambda_=args.lambda_, rand_samp=args.rand_samp)
    if args.nsamples < 1:
        args.nsamples = len(vis)
    for i in tqdm(range(args.nsamples)):
        if args.export_gt:
            pd_img, gt_img, imgpath = vis[i]
        else:
            pd_img, imgpath = vis[i]
        img_path = imgpath.split('/')[-1].split('.')[0]
        pd_save = os.path.join(args.output_dir, img_path+'_PD.png')
        plt.imsave(pd_save, pd_img)
        if args.export_gt:
            gt_save = os.path.join(args.output_dir, img_path+'_GT.png')
            plt.imsave(gt_save, gt_img)

if __name__ == '__main__':
    main()

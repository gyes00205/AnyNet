import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
import utils.logger as logger
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torchvision.transforms as transforms
import models.anynet
from PIL import Image


parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/media/bsplab/62948A5B948A3219/data_scene_flow_2015/testing/',
                    help='datapath')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--with_refine', action='store_true', help='with refine')
parser.add_argument('--output_dir', type=str, default='output', help='output dir')
parser.add_argument('--loadmodel', type=str, default='results/finetune_anynet_refine/checkpoint.tar', help='checkpoint')
args = parser.parse_args()

if args.datatype == '2015':
   from dataloader import KITTI_submission_loader as ls
elif args.datatype == '2012':
   from dataloader import KITTI_submission_loader2012 as ls  
elif args.datatype == 'other':
    from dataloader import diy_dataset as ls

test_left_img, test_right_img = ls.dataloader(args.datapath)

model = models.anynet.AnyNet(args)
os.makedirs(args.output_dir, exist_ok=True)
model = nn.DataParallel(model).cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
    model.eval()

    
    imgL = imgL.cuda()
    imgR = imgR.cuda()

    with torch.no_grad():
        output = model(imgL,imgR)[-1]
    output = torch.squeeze(output).data.cpu().numpy()
    return output

def main():
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(**normal_mean_var)]
    )    
    index_pbar = tqdm(range(len(test_left_img)))
    for inx in index_pbar:

        imgL_o = Image.open(test_left_img[inx]).convert('RGB')
        imgR_o = Image.open(test_right_img[inx]).convert('RGB')

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)         

        # pad to width and hight to 16 times
        if imgL.shape[1]%16 != 0:
            times = imgL.shape[1] // 16       
            top_pad = (times+1)*16 - imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2] // 16                       
            right_pad = (times+1)*16 - imgL.shape[2]
        else:
            right_pad = 0    

        imgL = F.pad(imgL,(0, right_pad, top_pad, 0)).unsqueeze(0)
        imgR = F.pad(imgR,(0, right_pad, top_pad, 0)).unsqueeze(0)

        start_time = time.time()
        pred_disp = test(imgL, imgR)
        index_pbar.set_description('time = %.2f' %(time.time() - start_time))

        if top_pad != 0 or right_pad != 0:
            img = pred_disp[top_pad:, :-right_pad]
        else:
            img = pred_disp

        img = (img*256).astype('uint16')
        img = Image.fromarray(img)
        img.save(os.path.join(args.output_dir, test_left_img[inx].split('/')[-1]))


if __name__ == '__main__':
    main()







import torch
from opt import opt
import os

from dataloader import ImageLoader

args = opt
args.dataset = 'coco'
# 
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    inputpath = args.inputpath
    inputlist = args.inputlist
    mode = args.mode

    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    if len(inputlist):
        im_names = open(inputlist,'r').readlines()
    elif len(inputpath) and inputpath != '/':
        for root,dirs,files in os.walk(inputpath):
            im_names = files
    else:
        raise IOError('Error: must contain either --indir/--list')

    ImageLoader("")



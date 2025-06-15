from dataloaders.video_list import get_loader
from dataset_path import Path
import os
from natsort import natsorted

def normal_dataloader(args):          
    train_loader = get_loader(dataset=args.dataset,
                              batchsize=args.batchsize,
                              size=args.trainsize,
                              split=args.trainsplit,
                              num_workers=args.threads, 
                              shuffle=True)
    val_loader = get_loader(dataset=args.dataset,
                            batchsize=args.batchsize,
                            size=args.trainsize,
                            split=args.valsplit,
                            num_workers=args.threads,
                            shuffle=False)
    print('Training with %d image pairs' % len(train_loader))
    print('Val with %d image pairs' % len(val_loader))
    return train_loader, val_loader 

def normal_test(args):   
    val_loader = get_loader(dataset=args.dataset,
                            batchsize=args.batchsize,
                            size=args.size,
                            split=args.split,
                            num_workers=args.threads,
                            shuffle=False)
    return val_loader 

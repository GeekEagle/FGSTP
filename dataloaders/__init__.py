from dataloaders.video_list import get_loader
from dataloaders.kfold_video import get_kfold_loader
from dataset_path import Path
import os
from natsort import natsorted

simgas_selection = [[0,  1,  2,  3,  4,  5, 6], 
                    [7,  8,  9,  10, 11, 12], 
                    [13, 14, 15, 16, 17, 18],
                    [19, 20, 21, 22, 23, 24],
                    [25, 26, 27, 28, 29, 30]]

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

def kfold_dataloader(args, iter):
    video_root = Path.db_root_dir(args.dataset)
    video_list = natsorted(os.listdir(video_root))
    val_list = [video_list[i] for i in simgas_selection[iter]]   
    train_list = list(set(video_list) - set(val_list))
    train_loader = None
    
    train_loader = get_kfold_loader(dataset=args.dataset,
                              batchsize=args.batchsize,
                              size=args.trainsize,
                              videos = train_list,
                              num_workers=args.threads, 
                              shuffle=True)
    val_loader = get_kfold_loader(dataset=args.dataset,
                            batchsize=args.batchsize,
                            size=args.trainsize,
                            videos = val_list,
                            num_workers=args.threads,
                            shuffle=False)
    return train_loader, val_loader, train_list, val_list

def kfold_test(args, iter):
    video_root = Path.db_root_dir(args.dataset)
    video_list = natsorted(os.listdir(video_root))
    val_list = [video_list[i] for i in simgas_selection[iter]]   
   
    val_loader = get_kfold_loader(dataset=args.dataset,
                            batchsize=args.batchsize,
                            size=args.testsize,
                            videos = val_list,
                            num_workers=args.threads,
                            shuffle=False)
    return val_loader, val_list

def normal_test(args):   
    val_loader = get_loader(dataset=args.dataset,
                            batchsize=args.batchsize,
                            size=args.size,
                            split=args.split,
                            num_workers=args.threads,
                            shuffle=False)
    return val_loader 
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from minefirst.lib.main_fgstp import FGSTP
from dataloaders import kfold_test    
from torchvision import transforms
import cv2, tqdm    

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='SimGas') 
parser.add_argument('--split', type=str, default='val')
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--trainsize', type=int, default=352, help='testing size')
parser.add_argument('--pretrained_cod10k', default='./pretrain/cod10k_encoder.pth', help='path to the pretrained Resnet')
parser.add_argument('--cuda', type=str, default='cuda:0', help='use cuda')
parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
opt = parser.parse_args()

def show_pictures(res, name, image_path, save_path, vis_path=None):
    res_pil = transforms.ToPILImage()(res)
    name = name.replace('jpg', 'png')
    frame_path = os.path.join(save_path, name)
    res_pil.save(frame_path)
    vis_file = os.path.join(vis_path, name)
    image_array = cv2.imread(image_path)
    res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    blend = (image_array/255 + res/255)*0.8
    blend = np.where(blend > 1, 1, blend)
    blend = (blend * 255).astype(np.uint8)  
    cv2.imwrite(vis_file, blend)

def compute_iou(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection, union

def f1_score(groundtruth, prediction):
    groundtruth = groundtruth.astype(bool)
    prediction = prediction.astype(bool)
    intersection = np.logical_and(groundtruth, prediction)
    recall = intersection.sum() / (groundtruth.sum() + 1e-8)
    precision = intersection.sum() / (prediction.sum() +1e-8)
    f1 = (2 * recall * precision) / (recall + precision + 1e-8)
    return f1

if __name__ == '__main__':
    save_root = './simgas_res'
    j_sum = 0
    f1_total = 0
    jf_sum = 0
    folds = 5
    for iter in range(folds):
        _, test_loader, _, test_list = kfold_test(opt, iter)
        model = FGSTP(opt).to(opt.cuda)
        pth_path = f'pretrain/{opt.dataset}/{opt.dataset}_{iter}_best.pth'

        model.load_state_dict(torch.load(pth_path))
        model.eval()
        iou_sum = 0
        union_sum = 0
        intersection_sum = 0
        f1_sum = 0
        confidence_sum = 0
        total_pictures = 0

        for i, data_blob in enumerate(test_loader):
            images, gt, scene, names, image_path = data_blob
            image = [img.to(opt.cuda) for img in images]
            _, _, pred = model(image)
            image_array = cv2.imread(image_path[0][0])
        
            for j in range(0, len(pred)):
                per_perd = pred[j].unsqueeze(0)
                res = F.upsample(per_perd, size=image_array.shape[:2], mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                pred_mask = np.where(res < 0.5, 0, res)
                prediction = np.where(res > 0.5, 1, 0)
                groundtruth = np.asarray(gt[j], np.float32)
                groundtruth /= (groundtruth.max() + 1e-8)
                groundtruth = np.where(groundtruth > 0.5, 1, 0)
                groundtruth = np.where(groundtruth > 0.5, 1, 0)
            
                groundtruth = groundtruth.squeeze()
                intersection, union = compute_iou(prediction, groundtruth)
                if union > 0:
                    total_pictures += 1
                    f1_sum += f1_score(groundtruth, prediction)
                    iou_sum += intersection / union
                
                prediction = (prediction * 255).astype(np.uint8)
                if iter < 5:
                    save_path = os.path.join(save_root, scene[j])
                    if not os.path.exists(save_path):
                       os.makedirs(save_path)
                    show_pictures(prediction, names[0][j], image_path[0][j], save_path)
        
        f1 = f1_sum / total_pictures
        Jaccard = iou_sum / total_pictures
        j_f = (f1 + Jaccard) / 2
        
        j_sum += Jaccard
        f1_total += f1
        jf_sum += j_f
        print('iter:{}, Jaccard: {}, f1: {}, J_F:{}.'.format(iter, Jaccard, f1, j_f))
    print('Jaccard: {}, f1: {}, J_F:{}.'.format(j_sum/folds, f1_total/folds, jf_sum/folds)) 
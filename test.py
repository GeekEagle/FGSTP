import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.main_fgstp import FGSTP
from dataloaders import normal_test
from torchvision import transforms
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='GasVid')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--size', type=int, default=352, help='testing size')
parser.add_argument('--trainsize', type=int, default=352, help='training size')
parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
parser.add_argument('--pth_path', type=str, default='./pretrain/gasvid_best.pth')
parser.add_argument('--pretrained_cod10k', default='./pretrain/cod10k_encoder.pth', help='path to the pretrained Resnet')
parser.add_argument('--cuda', type=str, default='cuda:0', help='use cuda? Default=True')
opt = parser.parse_args()

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name) / 1e6

def show_pictures(res, name, vis_path, save_path, image_path):
    res_pil = transforms.ToPILImage()(res)
    name = name.replace('jpg', 'png')
    mask_path = os.path.join(save_path, name)
    res_pil.save(mask_path)
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
    recall = intersection.sum() / (groundtruth.sum() +1e-8)
    precision = intersection.sum() / (prediction.sum() +1e-8)
    f1 = (2 * recall * precision) / (recall + precision + 1e-8)
    return f1

if __name__ == '__main__':
    test_loader = normal_test(opt)
    save_root = './gasvid_res'
    model = FGSTP(opt).to(opt.cuda)

    model.load_state_dict(torch.load(opt.pth_path))
    model.eval()

    # compute parameters
    print('Total Params = %.2fMB' % count_parameters_in_MB(model))
    iou_sum = 0
    f1_sum = 0
    total_pictures = 0
    
    with torch.no_grad():  
        for i, data_blob in enumerate(tqdm(test_loader)):
            images, gt, scene, names, image_path = data_blob
            image = [img.to(opt.cuda) for img in images]
            _, _, pred = model(image)
            image_array = cv2.imread(image_path[0][0])

            for j in range(0, len(pred)):
                per_perd = pred[j].unsqueeze(0)
                res = F.upsample(per_perd, size=image_array.shape[:2], mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                prediction = np.where(res > 0.5, 1, 0)
                gt_m = np.asarray(gt[j], np.float32)
                gt_m /= (gt_m.max() + 1e-8)
                groundtruth = np.where(gt_m > 0.5, 1, 0)
            
                groundtruth = groundtruth.squeeze()
                intersection, union = compute_iou(prediction, groundtruth)
                if np.sum(groundtruth) > 0:
                    total_pictures += 1
                    f1_sum += f1_score(groundtruth, prediction)
                    iou_sum += intersection / union            
                
                prediction = (prediction * 255).astype(np.uint8)
                
                save_path = os.path.join(save_root, scene[j], 'Masks')
                vis_path = os.path.join(save_root, scene[j], 'Vis')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    os.makedirs(vis_path)
                show_pictures(prediction, names[0][j], vis_path, save_path, image_path[0][j])
        
    f1 = f1_sum / total_pictures
    Jaccard = iou_sum / total_pictures
    j_f = (f1 + Jaccard) / 2
    print('ISG test, Jaccard: {}, f1: {}, J_F:{}.'.format(Jaccard, f1, j_f))


        

import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import sys
import time
import torch.nn as nn
import torch.optim as optim
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_CSANet import CSANet_dataset, RandomGenerator
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from scipy.ndimage import zoom
import wandb
import SimpleITK as sitk
import shutil



parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=224, help='Input patch size of network input')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--max_iterations', type=int, default=300000)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
parser.add_argument('--vit_patches_size', type=int, default=16)
parser.add_argument('--n_skip', type=int, default=3)
parser.add_argument('--number_of_folds', type=int, default=3)
args = parser.parse_args()


args.volume_path = '../data'
args.dataset = 'CSANet'
args.list_dir = './lists'
base_model_dir = "../model"
base_result_dir = "./probmaps"
combined_prob_map_dir = "./probmaps_combined"
final_nifti_output_dir = "./Results"
test_volume_dir = "../data/testVol"

patch_size = [args.img_size, args.img_size]



def create_probmaps(image_next, image, image_prev, net, classes, patch_size, case=None, foldnr=None):
    image = image.squeeze(0).cpu().detach().numpy()
    image_next, image_prev = image_next.squeeze(0).cpu().detach().numpy(), image_prev.squeeze(0).cpu().detach().numpy()
    out_prob_full = []

    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        slice_prev = image_prev[ind, :, :]
        slice_next = image_next[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]

        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
            slice_prev = zoom(slice_prev, (patch_size[0] / x, patch_size[1] / y), order=3)
            slice_next = zoom(slice_next, (patch_size[0] / x, patch_size[1] / y), order=3)

        input_curr = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        input_prev = torch.from_numpy(slice_prev).unsqueeze(0).unsqueeze(0).float().cuda()
        input_next = torch.from_numpy(slice_next).unsqueeze(0).unsqueeze(0).float().cuda()

        net.eval()
        with torch.no_grad():
            outputs = net(input_prev, input_curr, input_next)
            out_prob = torch.softmax(outputs, dim=1).squeeze(0).cpu().detach().numpy()
            out_prob_full.append(out_prob)

    results_fold_path = os.path.join(base_result_dir, f'fold_{foldnr}')
    os.makedirs(results_fold_path, exist_ok=True)
    file_path = os.path.join(results_fold_path, case[0] + '_probmap')
    np.save(file_path, out_prob_full)

    return out_prob


def probmaps_func(args, model, foldnr):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, require_labels=False)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    model.eval()
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, case_name = sampled_batch["image"], sampled_batch['case_name']
        image_next, image_prev = sampled_batch['next_image'], sampled_batch['prev_image']
        create_probmaps(image_next, image, image_prev, model, classes=args.num_classes,
                        patch_size=patch_size, case=case_name, foldnr=foldnr)




if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

dataset_config = {
    'CSANet': {
        'Dataset': CSANet_dataset,
        'volume_path': args.volume_path,
        'list_dir': args.list_dir,
        'num_classes': 4,
        'z_spacing': 1,
    }
}

args.num_classes = dataset_config[args.dataset]['num_classes']
args.Dataset = dataset_config[args.dataset]['Dataset']
args.exp = f"{args.dataset}_{args.img_size}"

snapshot_path = os.path.join(base_model_dir, args.exp, 'TU')
snapshot_path += f"_{args.vit_name}_skip{args.n_skip}"
if args.vit_patches_size != 16:
    snapshot_path += f"_vitpatch{args.vit_patches_size}"
if args.max_iterations != 30000:
    snapshot_path += f"_{str(args.max_iterations)[:2]}k"
if args.max_epochs != 30:
    snapshot_path += f"_epo{args.max_epochs}"
snapshot_path += f"_bs{args.batch_size}"
if args.base_lr != 0.01:
    snapshot_path += f"_lr{args.base_lr}"

config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
if 'R50' in args.vit_name:
    config_vit.patches.grid = (args.img_size // args.vit_patches_size, args.img_size // args.vit_patches_size)

net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()


for fold in range(5):
    model_path = os.path.join(snapshot_path, f'best_model_fold_{fold}.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(snapshot_path, f'epoch_{args.max_epochs - 1}.pth')
    net.load_state_dict(torch.load(model_path))

    log_folder = f'./test_log/test_log_{args.exp}'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_folder, f"fold{fold}.txt"),
                        level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    probmaps_func(args, net, foldnr=fold)


if os.path.exists(combined_prob_map_dir):
    for f in os.listdir(combined_prob_map_dir):
        os.remove(os.path.join(combined_prob_map_dir, f))
else:
    os.makedirs(combined_prob_map_dir)

fold0_dir = os.path.join(base_result_dir, 'fold_0')
for file in os.listdir(fold0_dir):
    if not file.endswith('.npy'):
        continue

    maps = [np.load(os.path.join(base_result_dir, f'fold_{i}', file)) for i in range(5)]
    mean_map = sum(maps) / len(maps)
    np.save(os.path.join(combined_prob_map_dir, file), mean_map)


if os.path.exists(final_nifti_output_dir):
    shutil.rmtree(final_nifti_output_dir)
os.makedirs(final_nifti_output_dir)

for image_file in os.listdir(combined_prob_map_dir):
    prob_map = np.load(os.path.join(combined_prob_map_dir, image_file))
    mask = np.argmax(prob_map, axis=1)

    case_id = "_".join(image_file.split("_")[:2])
    nii_path = os.path.join(test_volume_dir, f"{case_id}.nii.gz")
    if not os.path.exists(nii_path):
        print(f"Volume niet gevonden: {nii_path}, overslaan...")
        continue

    vol_image = sitk.ReadImage(nii_path)
    x, y = vol_image.GetSize()[1], vol_image.GetSize()[0]

    if x != patch_size[0] or y != patch_size[1]:
        pred = zoom(mask, (1, x / patch_size[0], y / patch_size[1]), order=0)
    else:
        pred = mask

    pred_img = sitk.GetImageFromArray(pred.astype(np.uint8))
    pred_img.SetSpacing(vol_image.GetSpacing())
    pred_img.SetDirection(vol_image.GetDirection())
    pred_img.SetOrigin(vol_image.GetOrigin())

    out_path = os.path.join(final_nifti_output_dir, f"{case_id}.nii.gz")
    sitk.WriteImage(pred_img, out_path)
    print(f"Opgeslagen: {out_path}")

 

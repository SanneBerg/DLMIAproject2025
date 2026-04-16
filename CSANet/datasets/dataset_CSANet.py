import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import SimpleITK as sitk
import os
import SimpleITK as sitk
from PIL import Image
import numpy as np
import cv2
from scipy.ndimage import rotate
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import scipy.ndimage


color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)

def apply_color_jitter(image):
    """Zorgt ervoor dat ColorJitter werkt op een numpy-array."""
    image_pil = Image.fromarray((image * 255).astype(np.uint8))  # Converteer naar PIL Image
    image_pil = color_jitter(image_pil)  # Pas de jitter toe
    return np.array(image_pil).astype(np.float32) / 255.0  # Converteer terug naar float numpy-array


def random_horizontal_flip(image,next_image, prev_image, segmentation):
    
    flip = random.choice([True, False])
    
    # Perform horizontal flipping if flip is True
    if flip:
        flipped_image = np.fliplr(image)
        flipped_next_image = np.fliplr(next_image)
        flipped_prev_image = np.fliplr(prev_image)
        flipped_segmentation = np.fliplr(segmentation)
    else:
        flipped_image = image
        flipped_next_image = next_image
        flipped_prev_image = prev_image
        flipped_segmentation = segmentation
    
    return flipped_image,flipped_next_image,flipped_prev_image,flipped_segmentation

def random_rotation(image, next_image, prev_image, label, angle_range=(-15, 15)):

    angle = random.uniform(angle_range[0], angle_range[1])  # Generate a random angle
    
    # Rotate each image and label
    rotated_image = rotate(image, angle, reshape=False, order=3, mode='nearest')
    rotated_next_image = rotate(next_image, angle, reshape=False, order=3, mode='nearest')
    rotated_prev_image = rotate(prev_image, angle, reshape=False, order=3, mode='nearest')
    rotated_label = rotate(label, angle, reshape=False, order=0, mode='nearest')
    
    return rotated_image, rotated_next_image, rotated_prev_image, rotated_label


def add_gaussian_noise(image, next_image, prev_image, label, mean=0, std=0.05):

    def apply_noise(img):
       
        noise = np.random.normal(mean, std, img.shape)
        noisy_img = img + noise
        return np.clip(noisy_img, 0, 1)  # Waarden binnen [0,1] houden

    return apply_noise(image), apply_noise(next_image), apply_noise(prev_image), label


def translate_image(image, shift_x=7, shift_y=0):

    rows, cols = image.shape
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])  # Transformatie matrix
    translated_image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return translated_image

def translate_x(image, next_image, prev_image, label, max_shift=7):
    
    shift_x = random.randint(-max_shift, max_shift)
    return (
        translate_image(image, shift_x, 0),
        translate_image(next_image, shift_x, 0),
        translate_image(prev_image, shift_x, 0),
        translate_image(label, shift_x, 0),
    )

def translate_y(image, next_image, prev_image, label, max_shift=7):
   
    shift_y = random.randint(-max_shift, max_shift)
    return (
        translate_image(image, 0, shift_y),
        translate_image(next_image, 0, shift_y),
        translate_image(prev_image, 0, shift_y),
        translate_image(label, 0, shift_y),
    )


    
def random_zoom(image, label, min_zoom=0.8, max_zoom=1.2):

    zoom_factor = random.uniform(min_zoom, max_zoom)  # Kies een willekeurige zoomschaal
    h, w = image.shape[:2]
    
    # Bereken de nieuwe afmetingen
    new_h = int(h * zoom_factor)
    new_w = int(w * zoom_factor)
    
    # Voer de zoomtransformatie uit
    zoomed_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    zoomed_label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)  # Gebruik NEAREST voor labels

    # Om te zorgen dat de zoom naar het oorspronkelijke formaat wordt gebracht, snijden we de afbeelding en label bij
    start_x = (new_w - w) // 2 if new_w > w else 0
    start_y = (new_h - h) // 2 if new_h > h else 0
    
    zoomed_image = zoomed_image[start_y:start_y + h, start_x:start_x + w]
    zoomed_label = zoomed_label[start_y:start_y + h, start_x:start_x + w]

    return zoomed_image, zoomed_label

def random_gaussian_blur(image, max_kernel_size=5):
    kernel_size = random.choice([3, 5, 7])
    if kernel_size <= max_kernel_size:
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return image


def random_brightness_contrast(image, lower=0.5, upper=1.5):
    alpha = random.uniform(lower, upper)  # Contrast factor
    beta = random.randint(-30, 30)  # Brightness factor
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image






class RandomGenerator(object):
    """
    Applies random transformations to a sample including horizontal flips and resizing to a target size.

    Parameters:
        output_size (tuple): Desired output dimensions (height, width) for the images and labels.
    """
    def __init__(self, output_size, mode = False):
        self.output_size = output_size
        self.mode = mode

    def __call__(self, sample):
        # Unpack the sample dictionary to individual components
        image, label = sample['image'], sample['label']
        next_image, prev_image = sample['next_image'], sample['prev_image']
        
#         image = normalize_image(image)
#         next_image = normalize_image(next_image)
#         prev_image = normalize_image(prev_image)
        
        # Apply a random horizontal flip to the images and label
        if self.mode:
            image,next_image, prev_image, label = random_horizontal_flip(image, next_image, prev_image, label)
            if random.random() < 0.50:  
                image,next_image, prev_image, label = random_rotation(image, next_image, prev_image, label)
            if random.random() < 0.50:  
                image, next_image, prev_image, label = translate_x(image, next_image, prev_image, label)
            if random.random() < 0.50:  
                image, next_image, prev_image, label = translate_y(image, next_image, prev_image, label)
            if random.random() < 0.50:  
                image, next_image, prev_image, label = add_gaussian_noise(image, next_image, prev_image, label)
            if random.random() < 0.50:  
                image = apply_color_jitter(image)
                next_image = apply_color_jitter(next_image)
                prev_image = apply_color_jitter(prev_image)
            if random.random() < 0.50:  # Apply zoom to both image and label
                image, label = random_zoom(image, label)  # Apply zoom to both the image and label
                next_image, _ = random_zoom(next_image, label)  # Apply zoom to next_image and label
                prev_image, _ = random_zoom(prev_image, label)  # Apply zoom to prev_image and label
            if random.random() < 0.50:  # Gaussian blur
                image = random_gaussian_blur(image)
                next_image = random_gaussian_blur(next_image)
                prev_image = random_gaussian_blur(prev_image)
            # if random.random() < 0.5:
            #     image, label = elastic_deformation(image, label)
            if random.random() < 0.5:
                image = random_brightness_contrast(image)
                next_image = random_brightness_contrast(next_image)
                prev_image = random_brightness_contrast(prev_image)
                
        image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_LINEAR)
        next_image = cv2.resize(next_image, self.output_size, interpolation=cv2.INTER_LINEAR)
        prev_image = cv2.resize(prev_image, self.output_size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.output_size, interpolation=cv2.INTER_NEAREST)  # Voor maskers is NEAREST beter
        
            


        # Check if the current size matches the desired output size
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            # Rescale images to match the specified output size using cubic interpolation
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            next_image = zoom(next_image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            prev_image = zoom(prev_image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            # Rescale the label using nearest neighbor interpolation (order=0) to avoid creating new labels
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # Convert numpy arrays to PyTorch tensors and add a channel dimension to images
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        next_image = torch.from_numpy(next_image.astype(np.float32)).unsqueeze(0)
        prev_image = torch.from_numpy(prev_image.astype(np.float32)).unsqueeze(0)
        # Return the modified sample as a dictionary
        sample = {'image': image, 'next_image': next_image, 'prev_image': prev_image, 'label': label.long()}
        return sample


def extract_and_increase_number(file_name):
    """
    Generates the filenames for the next and previous sequence by incrementing and decrementing the numerical part of a given filename.

    Parameters:
        file_name (str): The original filename from which to derive the next and previous filenames. 
                         The filename must end with a numerical value preceded by an underscore.

    Returns:
        tuple: Contains two strings, the first being the next filename in sequence and the second 
               the previous filename in sequence. If the original number is 0, the previous filename 
               will also use 0 to avoid negative numbering.
    """
    parts = file_name.rsplit("_", 1)
    parts_next = parts[0]
    parts_prev = parts[0]
    number = int(parts[1])
    
    next_number = number + 1
    prev_number = number - 1
    if prev_number== -1:
        pre_number = 0
    
    next_numbers = str(next_number)
    prev_numbers = str(prev_number)
    next_file_name = parts_next+"_"+str(next_numbers)
    prev_file_name = parts_prev+"_"+str(prev_numbers)

    return next_file_name,prev_file_name    
    
    
    
def check_and_create_file(file_name, image_name, folder_path):
    file_path = os.path.join(folder_path, "trainingImages", file_name+'.npy')
    if os.path.exists(file_path):
        return file_name
    else:
        available_name = image_name
        return available_name 


class CSANet_dataset(Dataset):
    """
    Dataset handler for CSANet, designed to manage image and mask data for training and testing phases.

    Attributes:
        base_dir (str): Directory where image and mask data are stored.
        list_dir (str): Directory where the lists of data splits are located.
        split (str): The current dataset split, indicating training or testing phase.
        transform (callable, optional): A function/transform to apply to the samples.

    Note:
        This class expects directory structures and file naming conventions that match the specifics
        given in the initialization arguments.
    """
    
    def __init__(self, base_dir, list_dir, split, transform=None,filter_patients=None,require_labels=True):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.image_sample_list = open(os.path.join(list_dir, 'train_image.txt')).readlines()
        if filter_patients:
            self.sample_list = [s for s in self.sample_list if s.split('_')[0] in filter_patients]
        
        self.mask_sample_list = open(os.path.join(list_dir, 'train_mask.txt')).readlines()
        if filter_patients:
            self.sample_list = [s for s in self.mask_sample_list if s.split('_')[0] in filter_patients]
        self.data_dir = base_dir
        self.require_labels = require_labels

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train_image" or self.split == "train_image_train" or self.split == "train_image_test":
            
            slice_name = self.image_sample_list[idx].strip('\n')
            image_data_path = os.path.join(self.data_dir, "trainingImages", slice_name+'.npy')
            image = np.load(image_data_path)
            #print("##################################### image path = ", image_data_path)
            # Manage sequence continuity by fetching adjacent slices
            next_file_name, prev_file_name = extract_and_increase_number(slice_name)
            
            next_file_name = check_and_create_file (next_file_name, slice_name, self.data_dir)
            prev_file_name = check_and_create_file (prev_file_name, slice_name, self.data_dir)
            
            
            next_image_path = os.path.join(self.data_dir, "trainingImages", next_file_name +'.npy')
            prev_image_path = os.path.join(self.data_dir, "trainingImages", prev_file_name +'.npy')
            
            next_image = np.load(next_image_path)
            prev_image = np.load(prev_image_path)           
            
            sample = {'image': image, 'next_image': next_image, 'prev_image': prev_image}
            
            if self.require_labels and self.mask_sample_list:
                mask_name = self.mask_sample_list[idx].strip('\n')
                label_data_path = os.path.join(self.data_dir, "trainingMasks", mask_name+'.npy')
                #print("############################################# label path = ", label_data_path)
                label = np.load(label_data_path)
                if os.path.exists(label_data_path):
                    label = np.load(label_data_path)
                else:
                    label = None  
                
                sample['label'] = label
        
            if self.transform:
                sample = self.transform(sample) # Apply transformations if specified
            sample['case_name'] = self.sample_list[idx].strip('\n')
            return sample
        else:
            # Handling testing data, assuming single volume processing
            vol_name = self.sample_list[idx].strip('\n')
            image_data_path = os.path.join(self.data_dir, "testVol", vol_name) #vergeet niet terug te zetten
            # image_data_path = os.path.join(self.data_dir, "testVol", vol_name) #vergeet niet terug te zetten
            # label_data_path = os.path.join(self.data_dir, "testMask", vol_name)
            if self.require_labels:
                label_data_path = os.path.join(self.data_dir, "testMask", vol_name.replace(".nii.gz", "_gt.nii.gz"))
            
            image_new = sitk.ReadImage(image_data_path) 
            img = sitk.GetArrayFromImage(image_new)
            
            
            next_image = sitk.GetArrayFromImage(image_new).astype(np.float64)
            prev_image = sitk.GetArrayFromImage(image_new).astype(np.float64)
            
            # Preprocess image data for testing phase
            combined_slices = sitk.GetArrayFromImage(image_new).astype(np.float64)
            
            
            for i in range(img.shape[0]):
                img_array = img[i, :, :].astype(np.uint8)
                p1 = np.percentile(img_array, 1)
                p99 = np.percentile(img_array, 99)

                normalized_img = (img_array - p1) / (p99 - p1)
                normalized_img = np.clip(normalized_img, 0, 1)

                combined_slices[i,:,:] = normalized_img
                
                if i-1 > -1 :
                    next_image[i-1,:,:] = combined_slices[i,:,:]
                
                if i-1<0:
                    prev_image[i,:,:] = combined_slices[i,:,:]
                else :
                    prev_image[i,:,:] = combined_slices[i-1,:,:]
            
            next_image[img.shape[0]-1,:,:] = combined_slices[img.shape[0]-1,:,:]
            sample = {'image': combined_slices, 'next_image': next_image, 'prev_image': prev_image}
            if self.require_labels:
                segmentation = sitk.ReadImage(label_data_path)
                label = sitk.GetArrayFromImage(segmentation)
                sample['label'] = label
            
            if self.transform:
                sample = self.transform(sample) # Apply transformations if specified
            num_string = self.sample_list[idx].strip('\n')
            case_num = num_string.split('.')[0]
            sample['case_name'] = case_num
            return sample

        

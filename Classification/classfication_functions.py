import os
import numpy as np
import glob
import nibabel as nib


class ClassificationACDC:
    def __init__(self):
        self.acdc_dict_train = []
        self.acdc_dict_test = []
        self.info_data = {}
        self.myocardium_density = 1.05                                     # in g/mL


    def read_patient_info(self, info_path):
        """Function that reads the relevant information from the patient_info .cfg files
        and returns this information in a dictionairy. These files contain information 
        about the disease, the height and the weight of the patient """
        
        info_data = {}
        
        # indicate which keys are extracted from the info file, this depends on the mode
        
        keys_to_extract = {'ED', 'ES', 'Group', 'Height', 'Weight'}
        

        # open the file and extract the data if possible
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                for line in f:
                    key_value = line.strip().split(":")
                    if len(key_value) == 2:
                        key, value = key_value
                        key = key.strip()
                        if key in keys_to_extract:
                            info_data[key] = value.strip()

        else:
            raise FileNotFoundError(f"{self.info_path} does not exist.")
        
        return info_data

    
    def build_class_dict_ACDC(self, segmentation_path, info_files_path, mode):
        """Function that builds a dictionairy with the information, the images and masks
        of each patient for both the ES and ED phase. This is needed for the training and testing of
        the Random Forrest Classifier. This function builds a dictionairy with the test data"""
        
        patients_path = glob.glob(os.path.join(info_files_path, '*.cfg'))
        
        for patient in patients_path:
            # get filename
            filename = os.path.basename(patient)
            # extract patient id
            patient_id = filename.split("_")[0]
            
            # read patient info with the help of the read_patient function
            patient_info = self.read_patient_info(patient)
            
            # extract the frames of the ES and ED phase 
            ED_frame = patient_info['ED']
            ES_frame = patient_info['ES']
            
            # make sure that the frames have the correct size 
            if len(ED_frame) == 1:
                ED_frame = '0'+ED_frame
            if len(ES_frame) == 1: 
                ES_frame = '0'+ES_frame

            if mode == 'test':   
                mask_path_ED = os.path.join(segmentation_path, f"{patient_id}_frame{ED_frame}_segmentation.nii.gz")
                mask_path_ES = os.path.join(segmentation_path, f"{patient_id}_frame{ES_frame}_segmentation.nii.gz")
            elif mode == 'train':
                mask_path_ED = os.path.join(segmentation_path, f"{patient_id}_frame{ED_frame}_gt.gz")
                mask_path_ES = os.path.join(segmentation_path, f"{patient_id}_frame{ES_frame}_gt.nii.gz")
            else:
                raise ValueError("Mode must be 'train' or 'test'")
                
            if os.path.exists(mask_path_ED) and os.path.exists(mask_path_ED):
                self.acdc_dict_test.append({'patient_info': {'Group': patient_info['Group'],'Height': patient_info['Height'], 'Weight': patient_info['Weight'] },
                            'ID': patient_id, 
                            'mask_ED': mask_path_ED,
                            'mask_ES': mask_path_ES
                            })
                
        return self.acdc_dict_test

    def create_dict_cardiac_info(self, mode):
            """Function to extract cardiac information based on the patient information and masks"""

            cardiac_info_list = []

            if mode == 'train': 
                acdc_dict = self.acdc_dict_train
            elif mode == 'test':
                acdc_dict = self.acdc_dict_test
            else:
                raise ValueError("Mode must be 'train' or 'test'")
                

            for sample_from_dict in acdc_dict:
                if 'mask_ES' in sample_from_dict:           # somehow last patient in dictionairy does not contain a ES-phase
                    # get info about volume ED 
                    mask_ED = nib.load(sample_from_dict['mask_ED'])
                    pixel_area_ED = mask_ED.header["pixdim"][1] * mask_ED.header["pixdim"][2]   # in mm^2
                    voxel_volume_ED = pixel_area_ED * mask_ED.header["pixdim"][3]               # in mm^3
                    
                    # seperate the information of each class 
                    msk_ed= mask_ED.get_fdata()
                    right_ventricle_ED = (msk_ed == 1)
                    myocardium_ED = (msk_ed == 2)
                    left_ventricle_ED = (msk_ed == 3)
                    
                    # calculate the volume of each class
                    RV_volume_ED = np.sum(right_ventricle_ED) * voxel_volume_ED * 1E-3    # in mL
                    LV_volume_ED = np.sum(left_ventricle_ED) * voxel_volume_ED * 1E-3     # in mL
                    MCD_volume_ED = np.sum(myocardium_ED) * voxel_volume_ED * 1E-3        # in mL
                    MCD_mass_ED = MCD_volume_ED * self.myocardium_density                 # in g
                    
                    # get info about volume ES
                    mask_ES = nib.load(sample_from_dict['mask_ES'])
                    pixel_area_ES = mask_ES.header["pixdim"][1] * mask_ES.header["pixdim"][2]   # in mm^2
                    voxel_volume_ES = pixel_area_ES * mask_ES.header["pixdim"][3]               # in mm^3
                    
                    # seperate the information of each class 
                    msk_es= mask_ES.get_fdata()
                    right_ventricle_ES = (msk_es == 1)
                    myocardium_ES = (msk_es == 2)
                    left_ventricle_ES = (msk_es == 3)
                    
                    # calculate the volume of each class
                    RV_volume_ES = np.sum(right_ventricle_ES) * voxel_volume_ES * 1E-3    # in mL
                    LV_volume_ES = np.sum(left_ventricle_ES) * voxel_volume_ES * 1E-3     # in mL
                    MCD_volume_ES = np.sum(myocardium_ES) * voxel_volume_ES * 1E-3        # in mL
                    MCD_mass_ES = MCD_volume_ES *self.myocardium_density                  # in g

                    # get the patient information 
                    patient_info = sample_from_dict['patient_info']

                    # calculate the ejection fraction and the bmi
                    EF_L = (LV_volume_ED-LV_volume_ES)/LV_volume_ED *100
                    BMI = float(patient_info['Weight'])/(float(patient_info['Height'])/100)**2
                 
                    cardiac_info_list.append({'Group': patient_info['Group']} | {'height':patient_info['Height'],
                            'weight': patient_info['Weight'],
                            'bmi': BMI,                                
                            'rv_volume_ed': RV_volume_ED, 
                            'lv_volume_ed': LV_volume_ED,
                            'myocardium_volume_ed': MCD_volume_ED, 
                            'myocardium_mass_ed': MCD_mass_ED,
                            'rv_volume_es': RV_volume_ES,
                            'lv_volume_es': LV_volume_ES, 
                            'myocardium_volume_es': MCD_volume_ES, 
                            'myocardium_mass_es': MCD_mass_ES,
                            'ejection_fraction_left': EF_L                   
                            })
            if mode == 'train': 
                self.cardiac_info_list_train = cardiac_info_list
                return self.cardiac_info_list_train

            elif mode == 'test':
                self.cardiac_info_list_test = cardiac_info_list
                return self.cardiac_info_list_test
    
     
    



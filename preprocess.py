import numpy as np
import SimpleITK as sitk
import glob
import os
import pandas as pd
from distutils.dir_util import copy_tree
import random
import nibabel as nib


def find_pair(name):
    #Outputs patient number, moving and fixed image scanID as strings for further analysis
    #Possible folder name pairs are below with each string differing in length
    # name = '10006_CL_Dev_004_CL_Dev_008'
    # name1 = 'CL_Dev_004_PS15_048'
    # name2 = 'PS15_048_CL_Dev_004'
    # name3 = 'PS15_048_PS17_017'

    sub_number = name[:5]
    
    #idx contains a list of strings of a given name
    
    idx = [s for s in name[6:].split("_")]


    if len(idx) == 6:
        
        mov = f'{idx[0]}_{idx[1]}_{idx[2]}'
        fix = f'{idx[3]}_{idx[4]}_{idx[5]}'
        return(sub_number, mov, fix)

    elif len(idx) == 5:
        if 'CL' in idx[0]:
            mov = f'{idx[0]}_{idx[1]}_{idx[2]}'
            fix = f'{idx[3]}_{idx[4]}'
            
            return(sub_number, mov, fix)
        elif 'PS' in idx[0]:
            mov = f'{idx[0]}_{idx[1]}'
            fix = f'{idx[2]}_{idx[3]}_{idx[4]}'
            
            return(sub_number, mov, fix)

    elif len(idx) == 4:
        mov = f'{idx[0]}_{idx[1]}'
        fix = f'{idx[2]}_{idx[3]}'
        return(sub_number, mov, fix)

    elif len(idx) == 3:
        mov = f'{idx[0]}'
        fix = f'{idx[1]}_{idx[2]}'
        return(sub_number, mov, fix)


    else:
        print('Not a corresponding folder name', name)

def transform_to_npz(data_path, df_path):
    
    df = pd.read_csv(df_path)
    
    img_paths = glob.glob(f'{data_path}*/*.nii.gz') 

    for img_path in img_paths:
        simg = sitk.ReadImage(img_path)
        npy_img = sitk.GetArrayFromImage(simg) #float 32
        # print(img_path)
        # npy_img = nib.load(img_path).get_fdata()
        print(npy_img.shape, npy_img[101,127,85], npy_img.dtype)

        path_elements = [s for s in img_path.split("/")]
        scan_id = path_elements[-1] 
        scan_id = scan_id.replace('.nii.gz', '')
        pair = path_elements[-2]
        sub_number, mov, fix = find_pair(pair)

        
        age = df[df['ScanID']==scan_id]['Age (Years)'].values[0]
        

        if not os.path.exists(f'/media/andjela/SeagatePor1/CP/npz_files/{sub_number}_{mov}_{fix}/'):
            os.mkdir(f'/media/andjela/SeagatePor1/CP/npz_files/{sub_number}_{mov}_{fix}/')
        # Assuming that you have 'age' and 'attribute' loaded, (add other attributes if necessary):
        np.savez_compressed(
            f'/media/andjela/SeagatePor1/CP/npz_files/{sub_number}_{mov}_{fix}/{scan_id}.npz',
            vol=npy_img,
            age=age,
        )   
def select_trainset(data_path, train_path):
    
    folders = glob.glob(f'{data_path}*')
    total_pairs = len(folders)
    # total_pairs = 434
    folders_train = folders[0:int(0.8*total_pairs)]
    print(total_pairs)
    print(len(folders_train))
    # print(folders_train)

    if not os.path.exists(train_path):
        os.mkdir(train_path)

    for i, folder in enumerate(folders_train):
        copy_tree(f'{folder}/', train_path)

def create_average_train(train_path, average_path):
    n = 100
    images = np.array(sorted(os.listdir(train_path)))
    sample = random.choices(images, k=n)
    

    image_array = []
    for image in sample:
        img_data = np.load(f"{train_path}{image}")["vol"]
        image = img_data[:, :, :, np.newaxis]
        # print('shape', image.shape) #188,229,229
        image_array.append(image)

    average = np.mean(np.array(image_array)[:,:,:,:,0], 0)
    
    if not os.path.exists(average_path):
        os.mkdir(average_path)

    np.savez_compressed(
            f'{average_path}linearaverage_100_train.npz',
            arr_0=average,
        )  

if __name__ == '__main__':

    data_path = '/media/andjela/SeagatePor1/CP/RigidReg_1.0/images/'
    csv_path = '/media/andjela/SeagatePor1/CP/Calgary_Preschool_Dataset_Updated_20200213_copy.csv'

    npz_path = '/media/andjela/SeagatePor1/CP/npz_files/'
    train_path = '/media/andjela/SeagatePor1/CP/npz_files/train/'
    average_path = '/media/andjela/SeagatePor1/CP/npz_files/averages/'
    # transform_to_npz(data_path=data_path, df_path=csv_path)
    # select_trainset(npz_path, train_path)
    create_average_train(train_path, average_path)

    # Save average image to nifti
    # img_data = np.load('/media/andjela/SeagatePor1/CP/npz_files/averages/linearaverage_100_train.npz')['vol']
    # nib_img = nib.Nifti1Image(img_data, None)
    # nib.save(nib_img, '/media/andjela/SeagatePor1/CP/npz_files/averages/linearaverage_100_train.nii.gz')
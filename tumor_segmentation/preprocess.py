import numpy as np
import nibabel as nib
import glob
import os


def normalize(img): #normlizáljuk a képet 0 és 1 közé
    img = img.astype(np.float32) #legyen csak 32bites az nibabel által visszadatto fp64,helyett (pytorch miatt)
    min_val = img.min()
    max_val = img.max()
    if max_val - min_val != 0:
        img = (img - min_val) / (max_val - min_val)
    return img

def load_nifti(path):
    return nib.load(path).get_fdata()

os.makedirs('BraTS2020_TrainingData/input_data_3channels/images', exist_ok=True) #kimeneti mappa létrehozása
os.makedirs('BraTS2020_TrainingData/input_data_3channels/masks', exist_ok=True)


root = 'MICCAI_BraTS2020_TrainingData/' #input fájlok helye

t2_list    = sorted(glob.glob(root + '*/*t2.nii'))#összegyűjtjük kölön listákba a különböző modalitásokat rendezve
t1ce_list  = sorted(glob.glob(root + '*/*t1ce.nii'))
flair_list = sorted(glob.glob(root + '*/*flair.nii'))
mask_list  = sorted(glob.glob(root + '*/*seg.nii'))


for idx in range(len(t2_list)):#bejárjuk az összes (lehetn t1ce is akár)
   # print(f"\n case {idx} ")

    
    t2    = normalize(load_nifti(t2_list[idx])) #különöbző modalitások betöltése nibabel és saját helper segítségével és normalizálás
    t1ce  = normalize(load_nifti(t1ce_list[idx]))
    flair = normalize(load_nifti(flair_list[idx]))

    mask = load_nifti(mask_list[idx]).astype(np.uint8) 
    mask[mask == 4] = 3       #brats labelek eredetelieg 0,1,2,4 de mi 0,1,2,3-at használunk

   
   
    img = np.stack([flair, t1ce, t2], axis=0) #a 3 összetartozó modalitást "egymásra pakoljuk" és létrehozunk egy 4D tömböt (Channel x Height x Width x Depth)

    img  = img[:, 56:184, 56:184, 13:141] #kivágjuk a középső 128 részt (a brats képek 240x240x155 méretűek eredetileg)
    mask = mask[56:184, 56:184, 13:141]

  
    _, counts = np.unique(mask, return_counts=True)
    if (1 - counts[0]/counts.sum()) < 0.01: #ha a maszkon az üres (0) voxelok aránya több mint 99% akkor kihagyjuk
        print(" → Skipped empty")
        continue


    np.save(f'BraTS2020_TrainingData/input_data_128/images/image_{idx}.npy', img) #npyt fájlok mentése
    np.save(f'BraTS2020_TrainingData/input_data_128/masks/mask_{idx}.npy', mask)



import numpy as np
import torch
from torch.utils.data import Dataset
import random

class BraTSDataset(Dataset): #saját Dataset osztály a BraTS előre preprocessált .npy fájlokhoz
    """
    Képek shape-je: (3, 128, 128, 128)
    Maszk shape:    (128, 128, 128)
    """

    def __init__(self, image_paths, mask_paths, augment=False, seed=None):
      
        assert len(image_paths) == len(mask_paths),  f"Hiba! Images: {len(image_paths)}, Masks: {len(mask_paths)}"  #maszkok és képek száma egyenlő-e?
          
        
        self.image_paths = image_paths  
        self.mask_paths = mask_paths
        self.augment = augment
       
        if seed is not None: #random seed beállítása reprodukálhatósághoz
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):#betöltés és augmentáció
      
        image = np.load(self.image_paths[idx]).astype(np.float32)   # (3,H,W,D)
        mask  = np.load(self.mask_paths[idx]).astype(np.int64)       # (H,W,D)
    
        image = torch.from_numpy(image)#a numpy tömbökat pytorch tenzorokká alakítjuk
        mask  = torch.from_numpy(mask) 
        
        if self.augment: # random flip 50%os eséllyel képek és maszkok is egyaránt
            
            if random.random() < 0.5:
                image = torch.flip(image, dims=[1])
                mask  = torch.flip(mask,  dims=[0])
            if random.random() < 0.5:
                image = torch.flip(image, dims=[2])
                mask  = torch.flip(mask,  dims=[1])
            if random.random() < 0.5:
                image = torch.flip(image, dims=[3])
                mask  = torch.flip(mask,  dims=[2])
        
        return image, mask
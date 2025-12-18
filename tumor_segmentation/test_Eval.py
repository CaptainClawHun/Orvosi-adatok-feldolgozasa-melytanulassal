import torch
import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt
from model import UNet3D_Light 

CHECKPOINT_PATH = r"checkpoints/best-epochepoch=91-diceval_dice=0.7123.ckpt" #tesztekls korábbi modellel
TEST_IMG_DIR = r"BraTS2021_Test_Data/input_data_test/images/"
TEST_MASK_DIR = r"BraTS2021_Test_Data/input_data_test/masks/"
SAVE_DIR = "evaluation_results"


def load_checkpoint(model, path, device):
  
    print(f"Súlyok betöltése: {path}")
    checkpoint = torch.load(path, map_location=device) #btöltjük a checkpointot
    

    if 'state_dict' in checkpoint: # dictionary pytorch lightning modellh kezeléshez
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint


    new_state_dict = {} #új state_dict létrehozása a "model." prefix eltávolításához eredetileg a model. benne van,de ez később nem kell
    for k, v in state_dict.items():
        name = k
        if name.startswith("model."):
            name = name[6:] 
        new_state_dict[name] = v

  
    try:
        model.load_state_dict(new_state_dict, strict=True) #betöltjük a súlyokat a modellbe
        print("Modell betöltve!")
    except Exception as e:
        print(f"Hiba: {e}")
        model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()#állítsuk értékelő módba
    return model


def calculate_dice(y_pred, y_true): #dice pontszám számítása teszthez (WT, TC, ET) 
    def d(a, b): return (2. * np.sum(a * b)) / (np.sum(a) + np.sum(b) + 1e-6)
    
    wt = d((y_pred > 0), (y_true > 0))
    tc = d(np.logical_or(y_pred==1, y_pred==3), np.logical_or(y_true==1, y_true==3))
    et = d((y_pred == 3), (y_true == 3))
    return wt, tc, et


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

   
    model = UNet3D_Light(in_channels=3, num_classes=4).to(device)
    model = load_checkpoint(model, CHECKPOINT_PATH, device) #modell betöltése

    images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.npy")))#bemeneti képek betöltése
    masks = sorted(glob.glob(os.path.join(TEST_MASK_DIR, "*.npy"))) #címkék betöltése
    

    print(f"Teszt: {len(images)} ")
    wt_scores, tc_scores, et_scores = [], [], [] #pontszámok tárolása

    with torch.no_grad():# nincs gradiens számítás
        for i in range(len(images)):# minden egyes teszt eset 3d mr képeken
            img = np.load(images[i])
            msk = np.load(masks[i])
            
           
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(device) #tensorrá alakítás és batch dimenzió hozzáadása
          
            pred = torch.argmax(model(img_tensor), dim=1).squeeze(0).cpu().numpy() #predkció készítése
            wt, tc, et = calculate_dice(pred, msk) #dice pontszám számítása
            wt_scores.append(wt); tc_scores.append(tc); et_scores.append(et) #pontszámok tárolása
            
            print(f"Case {i}: WT={wt:.3f} TC={tc:.3f} ET={et:.3f}")

    print("\nÖsszesített eredmények:")
    print(f"Átlag WT: {np.mean(wt_scores):.4f}")
    print(f"Átlag TC: {np.mean(tc_scores):.4f}")
    print(f"Átlag ET: {np.mean(et_scores):.4f}")
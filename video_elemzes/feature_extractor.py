import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

CONFIG = { # Konfigurációs beállítások
    'root_dir': r'C:/Users/Lenovo/Documents/BME/VII.Felev/szakdolgozat_final/video_elemzes/CholecT50/videos',
    'batch_size': 64,
    'image_size': 224,
    'model_name': 'resnet50' 
}

class FrameDataset(Dataset): #saját adatbetöltő osztály
    def __init__(self, video_path, transform):
       
        self.transform = transform 
        self.data = [] # adatok tárolása képkocka+hozzárendelt fázis
        
        vid_id = os.path.basename(video_path) # videó azonosító
        csv_filename = f"{vid_id}_phases.csv" # fájl neve
        label_file = os.path.join(video_path, csv_filename) # teljes elérési út
        
        if not os.path.exists(label_file): return # ha nincs fájl, kilépünk
        
        with open(label_file, 'r') as f: # fájl megnyitása
            lines = f.readlines()# sorok beolvasása
        if len(lines) > 0 and ("frame" in lines[0] or "frame_id" in lines[0]):# fejléc eltávolítása
            lines = lines[1:]
            
        for line in lines:#framek feldolgozása

            parts = line.strip().split(',')
            if len(parts) < 2: continue
            frame_idx = parts[0].strip() # frame index kinyerése
            try: phase_id = int(parts[1].strip()) #az id legyen int
            except ValueError: continue
            
            img_name = f"{int(frame_idx):06d}.png" #0-kal kitöltött fájlnév
            img_path = os.path.join(video_path, img_name)
            if os.path.exists(img_path):
                self.data.append((img_path, phase_id)) # hozzáadás a listához
    
        def __len__(self): return len(self.data) #adathalmaz mérete frame+fázis label
        
        def __getitem__(self, idx):
            path, label = self.data[idx] # elérési út és címke
            img = Image.open(path).convert('RGB') # kép betöltése
            img = self.transform(img) # transzformáció alkalmazása 
            
            return img, label

def get_transform(mode='default'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalizálás imageNet statisztikák alapján
    common = [transforms.Resize((CONFIG['image_size'], CONFIG['image_size']))] # közös átméretezés
    
    if mode == 'default':  #augmentációk alkalmazása (végül csak 1 volt alkalmazva )
        pass
    elif mode == 'aug1': # Szín + kis forgatás
        common.extend([
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.05,0.05))
        ])
    elif mode == 'aug2': # Tükrözés + Zaj
        common.extend([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.GaussianBlur(kernel_size=3)
        ])
    elif mode == 'aug3': # Erősebb szín + Kivágás
        common.extend([
            transforms.ColorJitter(saturation=0.2, hue=0.05),
            transforms.RandomResizedCrop(CONFIG['image_size'], scale=(0.9, 1.0))
        ])
        
    common.append(transforms.ToTensor())
    common.append(normalize)
    return transforms.Compose(common)

def extract_features():#fícsör iksztreksön
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("ResNet50 betöltése")
    resnet = models.resnet50(pretrained=True) # előre betanított modell betöltése
    model = nn.Sequential(*list(resnet.children())[:-1]).to(device) # levágjuk az utolsó réteget
    model.eval()# kiértékelő módba állítás
    
    video_folders = sorted([d for d in os.listdir(CONFIG['root_dir']) if d.startswith('VID')]) # videó mappák listázása
    print(f"Összesen {len(video_folders)} videó mappa.")
    
    # 4 verziót csinálunk: 1 eredeti + 3 augmentált ebből végül csak 1 augmentált lett használva
    modes = ['default', 'aug1', 'aug2', 'aug3']
    
    for mode in modes:
        print(f"\nProcessing mode: {mode.upper()} ")
        transform = get_transform(mode)
        
        # utótag a fájlnevekhez
        suffix = "" if mode == 'default' else f"_{mode}" #aug1aug2 aug3 stb
        
        for vid_id in tqdm(video_folders): #bejárjuk a videókat
            video_path = os.path.join(CONFIG['root_dir'], vid_id) # teljes elérési út
            
            
            feat_file = os.path.join(video_path, f'features_resnet50{suffix}.npy') #fícsör fájl elérési út a képekhe
            label_file_out = os.path.join(video_path, f'labels_resnet50{suffix}.npy') #fázisok
            
            if os.path.exists(feat_file): continue # Skip ha kész
            
            dataset = FrameDataset(video_path, transform) #adathalmaz létrehozása
            if len(dataset) == 0: continue #ha nincs adat, lépünk
            
            loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0) #adatbetöltő inicializálása
            
            feats, labels = [], [] #fícsörök és címkék tárolása
            with torch.no_grad():# kiértékelő mód, gradiens számítás kikapcsolva
                for x, y in loader:
                    x = x.to(device)
                    out = model(x).view(x.size(0), -1) #minden képre lesz egy 2048 hosszú  jellemző vektor
                    feats.append(out.cpu().numpy())#numpy tömbbé alakítás és tárolás
                    labels.append(y.numpy())#listába a címkéket is
            
            if feats:
                np.save(feat_file, np.concatenate(feats)) #fícsörök mentése fájlba
                np.save(label_file_out, np.concatenate(labels))#címkék mentése fájlba

if __name__ == "__main__":
    extract_features()
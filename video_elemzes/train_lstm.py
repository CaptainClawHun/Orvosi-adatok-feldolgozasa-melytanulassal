import os
import random
import torch
import torch.nn as nn
import numpy as np
import csv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report


CONFIG = { # Konfigurációs beállítások
    'root_dir': r'C:/Users/Lenovo/Documents/BME/VII.Felev/szakdolgozat_final/video_elemzes/CholecT50/videos',
    'sequence_length': 30, # LSTM bemeneti szekvencia hossza 30 frame
    'batch_size': 32,
    'num_epochs': 30,
    'learning_rate': 1e-4,
    'hidden_size': 128,
    'num_layers': 1,
    'bidirectional': True,
    'dropout': 0.25,
    'val_ratio': 0.2,
    'seed': 44,
    'label_smoothing': 0.1,
    'csv_name': 'training_log_original_2.csv',
    'cm_name': 'confusion_matrix_original_2.png'
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
random.seed(CONFIG['seed'])


def calculate_class_weights(root_dir, video_ids): #osztály súlyok számítása
    print("Osztályszámítás")#fontos mert nagyon kiegyesnsúlyozatlan az adathalmaz
    all_labels = []
    for vid_id in video_ids:
        p = os.path.join(root_dir, vid_id, 'labels_resnet50.npy')
        if os.path.exists(p):
            all_labels.extend(np.load(p)) #összes címke betöltése

    counts = Counter(all_labels)
    num_classes = max(counts.keys()) + 1 #7 osztál
    total = len(all_labels) #összes címke száma

    weights = [
        total / (num_classes * counts.get(i, 1)) #inverz súlyozás: amiből sok van az kapjon kisebb súlyt
        for i in range(num_classes)
    ]
    return torch.FloatTensor(weights)



def focal_loss(logits, targets, alpha=None, gamma=2.0 ,smoothing=0.0):
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction='none', label_smoothing=smoothing) # sima cross entropy
    pt = torch.exp(-ce)#
    return ((1 - pt) ** gamma * ce).mean() #focal loss számítás képlet alapján



class FeatureDataset(Dataset):
    def __init__(self, root_dir, vids, seq_len=30, mode='train'): 
        self.mode = mode
        self.seq_len = seq_len
        self.samples = []# (videó id, index) párok tárolása
        self.cache = {}#gyorsítótár

        for vid in vids:
            fpath = os.path.join(root_dir, vid, 'features_resnet50.npy') #fileok elérési útjai
            lpath = os.path.join(root_dir, vid, 'labels_resnet50.npy')#fázisok elérési útja
            if not os.path.exists(fpath):
                continue

            self.cache[vid] = {
                'features': np.load(fpath), #memóriába töltés
                'labels': np.load(lpath)
            }

            total = len(self.cache[vid]['labels'])
            if total < seq_len: #van-e elég képkocka a videóban egyáltalán?
                continue

            for idx in range(seq_len, total): #csúszó ablak
                self.samples.append((vid, idx))

      #  print(f"{mode.capitalize()} samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        vid, end = self.samples[i]
        cached = self.cache[vid]

        seq = cached['features'][end - self.seq_len:end] #szekvencia kinyerés
        label = cached['labels'][end - 1] #utolsó frame fázisa

        if self.mode == 'train':
            seq = seq + np.random.normal(0, 0.005, seq.shape)#zaj hozzáadás tréninghez

        return torch.FloatTensor(seq.copy()), torch.tensor(label, dtype=torch.long) #tenzor visszaadása



class BottleneckBiLSTM(nn.Module): #bidirekcionális LSTM modell saját
    def __init__(self, input_size, hidden, num_classes, drop):
        super().__init__()

        self.bottle = nn.Sequential( #dimenzió csökkentő réteg
            nn.Linear(input_size, 256),#2048-ból 256-ba csökkentés 
            nn.ReLU(),
            nn.Dropout(drop)#dropout a túltanulás ellen
        )

        self.lstm = nn.LSTM( #bidirekcionális LSTM réteg
            input_size=256,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden * 2, num_classes)# kimeneti réteg (2 mert bidirekcionális)

    def forward(self, x):
        x = self.bottle(x)
        out, _ = self.lstm(x)#az utolsó timestep kimenete (h_n és c_n nem kell)
        return self.fc(out[:, -1, :]) #elég csak az utolsó időlépés kimenete, mert az már látta az összes előző framet 



def plot_confusion_matrix(model, loader, device, save_path):
    model.eval()# kiértékelő mód
    preds, labels = [], []# predikciók és címkék tárolása

    with torch.no_grad():# gradiens számítás kikapcsolva
        for x, y in loader:
            x, y = x.to(device), y.to(device)# adat átvitele eszközre
            out = model(x)
            _, p = torch.max(out, 1)# legvalószínűbb osztály kiválasztása
            preds.extend(p.cpu().numpy())#két listába tároljuk a becsült és valós labeleket
            labels.extend(y.cpu().numpy())

    cm = confusion_matrix(labels, preds) #konfúziós mátrix számítása
    classes = sorted(list(set(labels)))

    plt.figure(figsize=(10, 8), dpi=150) #ábra létrehozása
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Best Model)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(classification_report(labels, preds))



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_vids = sorted([d for d in os.listdir(CONFIG['root_dir']) if "VID" in d])# videó mappák listázása
    random.shuffle(all_vids)# véletlenszerű sorrend (videóknál, nem pedig frameknél)

    split = int(len(all_vids) * (1 - CONFIG['val_ratio'])) #train-val felosztás
    train_vids = all_vids[:split]
    val_vids = all_vids[split:]

    train_ds = FeatureDataset(CONFIG['root_dir'], train_vids, CONFIG['sequence_length'], mode='train')
    val_ds = FeatureDataset(CONFIG['root_dir'], val_vids, CONFIG['sequence_length'], mode='val')

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    num_classes = max(np.concatenate([np.load(os.path.join(CONFIG['root_dir'], vid, 'labels_resnet50.npy')) for vid in all_vids])) + 1# osztályok száma
    class_w = calculate_class_weights(CONFIG['root_dir'], train_vids).to(device) #osztály súlyok kiszá,mítása

    model = BottleneckBiLSTM(2048, CONFIG['hidden_size'], num_classes, CONFIG['dropout']).to(device) #modell inicializálása

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4) #Adam optimalizáló
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3) #csökkenti a tanulási rátát ha nem javul a validációs pontosság
    criterion = lambda logits, labels: focal_loss(
    logits,
    labels,
    alpha=class_w,
    gamma=2.0,
    smoothing=CONFIG['label_smoothing']
    )

    best_acc = 0.0


    with open(CONFIG['csv_name'], 'w', newline='') as f:
        csv.writer(f).writerow(["Epoch", "TrainLoss", "TrainAcc", "ValLoss", "ValAcc", "LR"]) #logolás 


    for epoch in range(CONFIG['num_epochs']): #tanítás
        model.train()#tréning mód
        train_loss, correct, total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad() #gradiens nullázás
            out = model(x) 
            loss = criterion(out, y) #veszteség számítás
            loss.backward()# backpropagation
            optimizer.step()# súlyfrissítés

            train_loss += loss.item()
            _, pred = out.max(1)  #az indexre van szükség
            correct += (pred == y).sum().item() #helyes predikciók száma
            total += y.size(0) 

        train_acc = 100 * correct / total
        train_loss /= len(train_loader)


        model.eval() #kiértékelő mód
        val_loss, vcorrect, vtotal = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                _, p = out.max(1)
                vcorrect += (p == y).sum().item()
                vtotal += y.size(0)

        val_acc = 100 * vcorrect / vtotal
        val_loss /= len(val_loader)
        lr = optimizer.param_groups[0]['lr'] 

        scheduler.step(val_acc)

     
        with open(CONFIG['csv_name'], 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, lr]) #logolás

        print(f"Epoch {epoch+1:02d} | TrainLoss {train_loss:.4f} Acc {train_acc:.2f}% | " #logolás konzolra
              f"ValLoss {val_loss:.4f} Acc {val_acc:.2f}% | LR {lr:.1e}")

  
        if val_acc > best_acc:#ha van jobb modell mint eddig az akkor mentsük el
            best_acc = val_acc
            checkpoint = { 
                'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_acc': best_acc
            }
            torch.save(checkpoint, "best_checkpoint.pth") 
            print("New best model saved")


    print("\nBest model for confusion matrix:")
    best = torch.load("best_checkpoint.pth", map_location=device)# betöltjük a legjobb modellt
    model.load_state_dict(best['model_state'])

    plot_confusion_matrix(model, val_loader, device, CONFIG['cm_name']) #konfúziós mátrix mentése
    print("Confusion matrix saved:", CONFIG['cm_name'])


if __name__ == "__main__":
    main()
    
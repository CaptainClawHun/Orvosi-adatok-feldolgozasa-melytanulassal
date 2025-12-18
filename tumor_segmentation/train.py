import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
from logger import MetricsLogger 
from dataset import BraTSDataset
from model import UNet3D_Light  
from dataloader import get_dataloaders 


def dice_score(pred, target, num_classes=4, smooth=1e-6): # dice pontszám számítása
    pred = torch.argmax(pred, dim=1) #a legvalószínűbb osztály kiválasztása
    
    dice_scores = []
    for c in range(num_classes): 
        pred_c = (pred == c).float() #bináris maszkok adott osztályra (pred a modell kimenete, target a valós címke)
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()#két halmaz metszete voxel szinten 
        union = pred_c.sum() + target_c.sum()# dice képlet alapján nevező számítása 
        
        if union > 0:
            dice = (2. * intersection + smooth) / (union + smooth) #dice képlet
            dice_scores.append(dice)
    
    return torch.stack(dice_scores).mean() if dice_scores else torch.tensor(0.0) #átlag dice pontszám visszaadása


class CombinedLoss(nn.Module): #kombnált veszteségfüggvény definiálása (crossentropy + dice loss)
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def dice_loss(self, pred, target, smooth=1e-6): #dice veszteség számítása

        pred = F.softmax(pred, dim=1) #valószínűségek kiszámítása oszttályonként

        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]) #one-hot kódolás  kimenet: (B, D, H, W, C) minden osztály kap egy csatornát ahol 1.0 az értrék
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float() #átalakítás (B, C, D, H, W)
        
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1) # (B, C, D*H*W)
        target_flat = target_one_hot.reshape(target_one_hot.shape[0], target_one_hot.shape[1], -1)# (B, C, D*H*W)
        
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2) #a 3. dimenzió mentén összeadás
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean() #dice veszteség visszaadása
    
    def forward(self, pred, target):
        ce_loss = self.ce(pred, target) #crossentropy veszteség számítása
        dice_loss = self.dice_loss(pred, target) #dice veszteség számítása
        return 0.5 * ce_loss + 0.5 * dice_loss



class BraTSSegmentor(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()

        self.save_hyperparameters()
        self.model = UNet3D_Light(in_channels=3, num_classes=4) #saját 3D U-Net modell
        self.loss_fn = CombinedLoss()#kombinált veszteségfüggvény
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        images, masks = batch #bemeneti képek és címkék
        logits = self(images)#modell előrejelzése
        
        loss = self.loss_fn(logits, masks)  #kombinált veszteség számítása
        dice = dice_score(logits, masks)# dice pontszám számítása
        
       
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_dice", dice, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss #visszatér a veszteséggel a backpropagationhoz
    
    def validation_step(self, batch):
        images, masks = batch
        logits = self(images)
        
        loss = self.loss_fn(logits, masks)
        dice = dice_score(logits, masks)
    
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_dice", dice, prog_bar=True, on_epoch=True)
        
        return {"val_loss": loss, "val_dice": dice}
    
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',
            factor=0.5,
            patience=5 #ha 5 epochon keresztül nem javul a val_dice, akkor csökkenti a tanulási rátát
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_dice"
            }
        }

def start_training():
 
    print("\nLoading data")
    train_loader, val_loader = get_dataloaders(
        data_dir=r"BraTS2020_TrainingData/input_data_128/",
        batch_size=1,
        num_workers=2,
        test_size=0.2,
        seed=42
    )

    checkpoint_callback = ModelCheckpoint( #modell mentése legjobb val_dice alapján
        dirpath="checkpoints/",
        monitor="val_dice",
        mode="max",
        filename="best-epoch{epoch:02d}-dice{val_dice:.4f}",
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    metrics_logger = MetricsLogger(save_dir='metrics/', filename='training_metrics.csv')
    logger = TensorBoardLogger("logs/", name="brats3d")
    
    print("\create trainer")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=100,
        precision=16, 
        callbacks=[checkpoint_callback, lr_monitor, metrics_logger],  
        logger=logger,
        log_every_n_steps=10,
        accumulate_grad_batches=4, 
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    print("\n Create model")
    model = BraTSSegmentor(learning_rate=1.1e-4)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    
    trainer.fit(model, train_loader, val_loader,ckpt_path="checkpoints/best-epochepoch=91-diceval_dice=0.7123.ckpt")
    

    print("Training Complete")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_dice: {checkpoint_callback.best_model_score:.4f}")
 
if __name__ == "__main__":
    start_training()
import torch
import torch.nn as nn

class DoubleConv(nn.Module): #dupla konvolúciós réteg egymás után +relu
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.step = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1), #3x3x3 kernel size
            nn.ReLU(inplace=True), #memóri optimalizálás
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.step(x)




class UNet3D_Light(nn.Module): #saját 3D-Unet (egyszerűsített változat)
    def __init__(self, in_channels=3, num_classes=4):
        super().__init__()
        
        
        self.layer1 = DoubleConv(in_channels, 16)   #encoder ág definiálása 3-> 16->32->64->128
        self.layer2 = DoubleConv(16, 32)            
        self.layer3 = DoubleConv(32, 64)            
        self.layer4 = DoubleConv(64, 128)           
        
       
        self.layer5 = DoubleConv(128 + 64, 64) #decoder ág és skip connections 3+5 
        self.layer6 = DoubleConv(64 + 32, 32) #2+6
        self.layer7 = DoubleConv(32 + 16, 16)#1+7
        self.layer8 = nn.Conv3d(16, num_classes, kernel_size=1)#kimenet
        
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2) #pooling 
        
    def forward(self, x):
        x1 = self.layer1(x) 
        x1m = self.maxpool(x1)
        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)
        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)
        x4 = self.layer4(x3m)
        x5 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)(x4) #növeljük a térbeli méretet 
        x5 = torch.cat([x5, x3], dim=1) #skip connection összefűzés
        x5 = self.layer5(x5)
        x6 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)(x5)   
        x6 = torch.cat([x6, x2], dim=1) 
        x6 = self.layer6(x6)
        x7 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.layer7(x7)
        output = self.layer8(x7)
        return output
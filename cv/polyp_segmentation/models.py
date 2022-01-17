import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
from torch.nn import functional as F

class ConsecutiveConvolution(nn.Module):
    def __init__(self,input_channel,out_channel):
        super(ConsecutiveConvolution,self).__init__()
        self.conv = nn.Sequential(
            
            nn.Conv2d(input_channel,out_channel,3,1,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(out_channel,out_channel,3,1,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True),            
        
        )
        
    def forward(self,x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self,input_channel, output_channel, features = [64,128,256,512,1024,2048]):
        super(UNet,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # initialize the encoder
        for feat in features:
            self.encoder.append(
                ConsecutiveConvolution(input_channel, feat)    
            )
            input_channel = feat
        
        #initialize the decoder 
        for feat in reversed(features):
            # the authors used transpose convolution
            self.decoder.append(nn.ConvTranspose2d(feat*2, feat, kernel_size=2, stride=2))
            self.decoder.append(ConsecutiveConvolution(feat*2, feat))
        
        #bottleneck
        self.bottleneck = ConsecutiveConvolution(features[-1],features[-1]*2)
        
        #output layer
        self.final_layer = nn.Conv2d(features[0],output_channel,kernel_size=1)
        
    def forward(self,x):
        skip_connections = []
        
        #encoding
        for layers in self.encoder:
            x = layers(x)
            #skip connection to be used in recreation 
            skip_connections.append(x)

            x = self.pool(x)
        
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        
        
        for idx in range(0,len(self.decoder),2):
            
            
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx//2]
            
    
            if x.shape != skip_connection.shape[2:]:
                x = TF.resize(x,size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection,x),dim=1)

            x = self.decoder[idx+1](concat_skip)
        
        return self.final_layer(x)
import satlaspretrain_models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import os 

from data_loader import *
from loss import *
from utils import *

NUM_CLASSES = 1
LEARNING_RATE = 0.01
EPOCHS = 50
IMAGE_SIZE = (256, 256)
N_CHANNELS = 10
BATCH_SIZE = 10
OUTPUT_DIR = '/home/marycamila/flaresat/train/train_output'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model():
    weights_manager = satlaspretrain_models.Weights()
    model = weights_manager.get_pretrained_model("Landsat_SwinB_SI", head=satlaspretrain_models.Head.BINSEGMENT, fpn=True, num_categories=NUM_CLASSES)
    model = model.to(device)
    return model

def freeze_layers(model):
    first_layer = model.backbone.backbone.features[0][0]
    model.backbone.backbone.features[0][0] = torch.nn.Conv2d(N_CHANNELS,
                                        first_layer.out_channels,
                                        kernel_size=first_layer.kernel_size,
                                        stride=first_layer.stride,
                                        padding=first_layer.padding,
                                        bias=(first_layer.bias is not None))

    for param in model.parameters():
        param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if 'head.layers' in name or 'upsample.layers' in name or 'fpn.fpn' in name:
    #         param.requires_grad = True
    #     if 'backbone.backbone.head' in name:
    #         param.requires_grad = True     
    
    for name, param in model.named_parameters():
        if 'head' in name:
            param.requires_grad = True
    
    return model

pretrained_model = load_model()
model = freeze_layers(pretrained_model)
model = model.to(device)

images_train = pd.read_csv('/home/marycamila/flaresat/dataset/images_train.csv')['tiff_file']
masks_train = pd.read_csv('/home/marycamila/flaresat/dataset/masks_train.csv')['mask_file']

images_validation = pd.read_csv('/home/marycamila/flaresat/dataset/images_val.csv')['tiff_file']
masks_validation = pd.read_csv('/home/marycamila/flaresat/dataset/masks_val.csv')['mask_file']

image_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1])
train_dataset = ImageGenerator(images_train, masks_train, N_CHANNELS, NUM_CLASSES, image_shape)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = ImageGenerator(images_validation, masks_validation, N_CHANNELS, NUM_CLASSES, image_shape)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# summarize_model(model)

criterion = BinaryFocalLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print('FlareSat - Train initiated')

best_loss = 10000
train_losses = []
val_losses = []

model.train()

for epoch in range(EPOCHS):
    print("Starting Epoch...", (epoch+1))
    running_loss = 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        
        output, _ = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader) 
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.12f}")

    # Validation
    model.eval() 
    val_running_loss = 0.0

    with torch.no_grad():
        for val_data, val_target in val_loader:
            val_data, val_target = val_data.to(device), val_target.to(device)

            val_output, _ = model(val_data)       
            val_loss = criterion(val_output, val_target)

            val_running_loss += val_loss.item()

    val_epoch_loss = val_running_loss / len(val_loader)
    val_losses.append(val_epoch_loss)
    print(f"Validation Loss: {val_epoch_loss:.12f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss

        transfer_learning_model = 'transfer_learning_flaresat.pth'

        model_dir = os.path.join(OUTPUT_DIR, transfer_learning_model)
        
        torch.save(model.state_dict(), model_dir)
        print(f"Model saved with loss: {epoch_loss:.12f}")

print('FlareSat - Train finished!')

# Plot loss progression
plt.plot(range(1, EPOCHS + 1), train_losses, label='training')
plt.plot(range(1, EPOCHS + 1), val_losses, label='validation')
plt.legend()
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(os.path.join(OUTPUT_DIR, "transfer_learning_loss.png"), dpi=300, bbox_inches='tight')
plt.clf()
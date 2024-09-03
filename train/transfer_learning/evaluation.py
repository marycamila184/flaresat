from sklearn.metrics import precision_score, recall_score, f1_score
import satlaspretrain_models
import tifffile as tiff
import torch
import numpy as np
import pandas as pd

CUDA_DEVICE = "cpu"
THRESHOLD = 0.50
IMAGE_SIZE = (256, 256)
N_CHANNELS = 10
NUM_CLASSES = 1

def get_img_arr(file_path, n_channels):
    img = tiff.imread(file_path)
    img = np.resize(img, (256, 256, 10))
    
    if n_channels == 10:
        img = img[:, :, :]
    elif n_channels == 3:
        # Active-fire 
        img = img[:, :, [1,5,6]]
        #img = img[:, :, [4,5,6]]
    elif n_channels== 2:
        img = img[:, :, [5,6]]

    return img

def get_mask_arr(file_path):
    mask = tiff.imread(file_path)
    mask = np.resize(mask, (256, 256))
    mask = np.float32(mask)/255

    return mask

device = torch.device(CUDA_DEVICE)
print(f"Using device: {device}")

weights_manager = satlaspretrain_models.Weights()
model = weights_manager.get_pretrained_model("Landsat_SwinB_SI", head=satlaspretrain_models.Head.SEGMENT, fpn=True, num_categories=NUM_CLASSES)
model = model.to(device)

first_layer = model.backbone.backbone.features[0][0]
model.backbone.backbone.features[0][0] = torch.nn.Conv2d(N_CHANNELS,
                                        first_layer.out_channels,
                                        kernel_size=first_layer.kernel_size,
                                        stride=first_layer.stride,
                                        padding=first_layer.padding,
                                        bias=(first_layer.bias is not None))

model_path = '/home/marycamila/flaresat/train/train_output/transfer_learning_flaresat.pth'
recent_weights = torch.load(model_path, map_location=device)
pretrained_model = model.load_state_dict(recent_weights)

images_test = pd.read_csv('/home/marycamila/flaresat/dataset/images_test.csv')
masks_test = pd.read_csv('/home/marycamila/flaresat/dataset/masks_test.csv')

test_images = np.array([get_img_arr(path, N_CHANNELS) for path in images_test['tiff_file']])
test_masks = np.array([get_mask_arr(path) for path in masks_test['mask_file']])

test_images_loader = torch.tensor(test_images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
test_masks_loader = torch.tensor(test_masks, dtype=torch.float32).unsqueeze(1).to(device)

precision_list, recall_list, f1_list, iou_list = [], [], [], []
n_images = len(test_images)
batch_size = 128

model.eval()

with torch.no_grad():
    for i in range(0, len(test_images), batch_size):
        end_idx = min(i + batch_size, n_images)

        batch_images = test_images[i:end_idx]
        batch_masks = test_masks[i:end_idx]
    
        batch_images_loader = torch.tensor(batch_images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        batch_masks_loader = torch.tensor(batch_masks, dtype=torch.float32).unsqueeze(1).to(device)

        y_pred_batch, _ = model(batch_images_loader)
        y_pred_batch = y_pred_batch.numpy()

        y_pred_binary_batch = (y_pred_batch > THRESHOLD).astype(int)

        y_test_flat = batch_masks_loader.numpy().flatten()
        y_pred_flat = y_pred_binary_batch.flatten()

        precision_list.append(precision_score(y_test_flat, y_pred_flat))
        recall_list.append(recall_score(y_test_flat, y_pred_flat))
        f1_list.append(f1_score(y_test_flat, y_pred_flat))

        intersection = np.logical_and(y_test_flat, y_pred_flat)
        union = np.logical_or(y_test_flat, y_pred_flat)
        iou_list.append(np.sum(intersection) / np.sum(union))

    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    f1 = np.mean(f1_list)
    iou = np.mean(iou_list)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"IoU: {iou}")
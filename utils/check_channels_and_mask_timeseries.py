import tifffile as tiff
from PIL import Image
import numpy as np

def read_and_save_tiff_as_png(file_path, mask_path, file, mask):
    patch_img = tiff.imread(file_path)
    patch_img = np.resize(patch_img, (256, 256, 10))
    patch_channel_data = patch_img[:, :, 6]
    patch_channel_data = 255 * patch_channel_data
    patch_channel_data = patch_channel_data.astype(np.uint8)
    patch_img_pil = Image.fromarray(patch_channel_data)
    
    mask_img = tiff.imread(mask_path)
    mask_img = np.resize(mask_img, (256, 256))
    mask_img = mask_img.astype(np.uint8)
    mask_img_pil = Image.fromarray(mask_img)
    
    combined_img = Image.new('RGB', (patch_img_pil.width + mask_img_pil.width, patch_img_pil.height))
    combined_img.paste(patch_img_pil, (0, 0))
    combined_img.paste(mask_img_pil, (patch_img_pil.width, 0))
    
    combined_output_path = f"{file}_and_{mask}.png"
    combined_img.save(combined_output_path)
    
    print(f"Saved {combined_output_path}")

list_image = ['LC81950372019237LGN00', 'LC81950372019221LGN00', 'LC81950372019269LGN00', 'LC81950372019253LGN00']
row_col_index = '21_12'
for scene in list_image:
    file_path = f'/home/marycamila/flaresat/dataset/flare_patches/fire_{scene}_{row_col_index}_patch.tiff'
    mask_path = f'/home/marycamila/flaresat/dataset/mask_patches/fire_{scene}_{row_col_index}_mask.tiff'
    file = f'patch_{scene}_{row_col_index}'
    mask = f'mask_{scene}_{row_col_index}'
    read_and_save_tiff_as_png(file_path, mask_path, file, mask)

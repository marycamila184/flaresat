import numpy as np
import cv2
import os

def plot_inferences(truth_masks, method_masks, truth_patches, cloud_masks, flaresat_masks, output_path, list_entities_plot, method, n_images=30):
    for index in range(n_images):
        original_image = truth_patches[index][:, :, 6]
        original_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX)
        original_image = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(original_image, 'B7 Landsat', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        truth_mask = truth_masks[index][:, :]
        truth_mask = cv2.normalize(truth_mask, None, 0, 255, cv2.NORM_MINMAX)
        truth_mask = cv2.cvtColor(truth_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(truth_mask, 'Ground Truth Mask', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cloud_mask = cloud_masks[index][:, :]
        cloud_mask = cv2.normalize(cloud_mask, None, 0, 255, cv2.NORM_MINMAX)
        cloud_mask = cv2.cvtColor(cloud_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(cloud_mask, 'Cloud Mask', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        flaresat_mask = flaresat_masks[index].squeeze()
        flaresat_mask = cv2.normalize(flaresat_mask, None, 0, 255, cv2.NORM_MINMAX)
        flaresat_mask = cv2.cvtColor(flaresat_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(flaresat_mask, 'Predicted Flare Mask', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        predicted_mask = method_masks[index][:, :]
        predicted_mask = predicted_mask.astype(np.uint8)
        predicted_mask = cv2.normalize(predicted_mask, None, 0, 255, cv2.NORM_MINMAX)
        predicted_mask = cv2.cvtColor(predicted_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        text_image = method
        if len(list_entities_plot) > 0:
            text_image += " " + list_entities_plot[index]
        cv2.putText(predicted_mask, str(text_image).upper() + ' Predicted Mask', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        concatenated_image = np.hstack((original_image, cloud_mask, truth_mask, predicted_mask, flaresat_mask))

        file_name = f"inference_{method}_{index}.png"
        save_path = os.path.join(output_path, file_name)
        cv2.imwrite(save_path, concatenated_image)


def plot_scene_and_squared(full_scene, scene, scene_id, row, col, output_path, method):
    # Process grayscale scene
    file_name_scene = f"scene_{method}_{scene_id}.png"
    save_path_scene = os.path.join(output_path, file_name_scene)

    scene = cv2.normalize(scene, None, 0, 255, cv2.NORM_MINMAX)
    scene = cv2.cvtColor(scene.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    square_top_left = (max(0, col - 25), max(0, row - 25))
    square_bottom_right = (min(scene.shape[1], col + 25), min(scene.shape[0], row + 25))

    cv2.rectangle(scene, square_top_left, square_bottom_right, (0, 255, 0), 2)
    cv2.imwrite(save_path_scene, scene)

    # Print B7 scene
    file_name_b7 = f"scene_B7_{method}_{scene_id}.png"
    save_path_b7 = os.path.join(output_path, file_name_b7)

    b7_band = cv2.normalize(full_scene[:, :, 6], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    b7_band = cv2.cvtColor(b7_band, cv2.COLOR_GRAY2BGR)

    square_top_left = (max(0, col - 25), max(0, row - 25))
    square_bottom_right = (min(b7_band.shape[1], col + 25), min(b7_band.shape[0], row + 25))

    cv2.rectangle(b7_band, square_top_left, square_bottom_right, (0, 255, 0), 2)
    cv2.imwrite(save_path_b7, b7_band)

    # Print COLORED scene
    file_name_scene_color = f"scene_colored_{method}_{scene_id}.png"
    save_path_scene_color = os.path.join(output_path, file_name_scene_color)

    if method == 'nhi':
        red_band = full_scene[:, :, 3]  # Band 4 (Red)
        green_band = full_scene[:, :, 2]  # Band 3 (Green)
        blue_band = full_scene[:, :, 1]  # Band 2 (Blue)
    else:
        red_band = full_scene[:, :, 6]  # Band 7 (SWIR)
        green_band = full_scene[:, :, 5]  # Band 6 (SWIR)
        blue_band = full_scene[:, :, 4]  # Band 5 (NIR)

    # Normalize each band separately to enhance contrast
    red_band = cv2.normalize(red_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    green_band = cv2.normalize(green_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blue_band = cv2.normalize(blue_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Stack the bands into a false-color composite
    scene_color = np.dstack((red_band, green_band, blue_band))
    scene_color = cv2.cvtColor(scene_color, cv2.COLOR_RGB2BGR)

    square_top_left_fc = (max(0, col - 25), max(0, row - 25))
    square_bottom_right_fc = (min(scene_color.shape[1], col + 25), min(scene_color.shape[0], row + 25))

    cv2.rectangle(scene_color, square_top_left_fc, square_bottom_right_fc, (0, 255, 0), 2)
    cv2.imwrite(save_path_scene_color, scene_color)
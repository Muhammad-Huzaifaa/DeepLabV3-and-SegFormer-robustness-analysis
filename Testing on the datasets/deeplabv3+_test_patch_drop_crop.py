from pixellib.semantic import semantic_segmentation
import cv2
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

segment_image = semantic_segmentation()
segment_image.load_ade20k_model("deeplabv3_xception65_ade20k.h5")


def IoU_coeff(y_true, y_pred):
    axes = (0,1) 
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask - intersection
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


## Iterating over the images and predicting the masks, then then storing them in a pridictions folder
def load_and_get_iou(img_dir, mask_dir, pred_mask_dir):
    segmap, seg_overlay = segment_image.segmentAsAde20k(img_dir, overlay = False)
    ## getting the prediction
    try:
        if 'person' in segmap['class_names'] and np.any(seg_overlay[np.array(segmap['class_colors'])[np.array(segmap['class_names'])=='person']==seg_overlay]): 
            predicted_mask = np.zeros(seg_overlay.shape)
            predicted_mask[np.array(segmap['class_colors'])[np.array(segmap['class_names'])=='person']==seg_overlay] = seg_overlay[np.array(segmap['class_colors'])[np.array(segmap['class_names'])=='person']==seg_overlay][0]
            predicted_mask=np.average(predicted_mask, axis = 2)
            predicted_mask*=(1.0/np.amax(predicted_mask))
            ##saving the predicted mask
            cv2.imwrite(pred_mask_dir, predicted_mask)
            ##getting the ground truth 
            ground_truth = cv2.imread(mask_dir, 0)
            ground_truth = np.float64(ground_truth)*(1.0/np.amax(ground_truth))
            iou_img = IoU_coeff(ground_truth, predicted_mask)
        else:
            iou_img = 0
    except:
        iou_img = 0
    return iou_img




def main():
    all_iou = {}
    for dir_name  in ["0.3", "0.5", "0.7"]:
        for aug_type in ["random_cropped","patch_cropped"]:
            iou_list = []
            print("starting ", dir_name, "/", aug_type)
            for root, dirs, files in os.walk("../people_segmentation/original/images/", topdown=False):
                pred_mask_dir = f"../people_segmentation/{dir_name}/{aug_type}/pred_masks"
                for img_name in tqdm(files):
                    img_name = img_name[:-4]
                    img_dir = f"../people_segmentation/{dir_name}/{aug_type}/images/{img_name}.jpg"
                    mask_dir = f"../people_segmentation/{dir_name}/{aug_type}/masks/{img_name}.png"
                    pred_mask_dir = f"../people_segmentation/{dir_name}/{aug_type}/deeplab_pred/{img_name}.png"
                    iou_list.append(load_and_get_iou(img_dir=img_dir, mask_dir=mask_dir, pred_mask_dir=pred_mask_dir))
                    
            all_iou[f'{aug_type}_{dir_name}'] = iou_list

    print(all_iou)
    df = pd.DataFrame.from_dict(all_iou)
    df.to_csv(f"../people_segmentation/all_iou_deeplab.csv")

main()


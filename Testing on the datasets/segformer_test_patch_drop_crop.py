from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
from sklearn.metrics import confusion_matrix  
import os
import torch
from torch import nn
import os
import pandas as pd
from tqdm import tqdm


def IoU_coeff(y_true, y_pred):
    axes = (0,1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask - intersection
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def load_and_get_iou(img_name ,model, feature_extractor, device, show_img = False, img_dir = "../people_segmentation/original/images", 
                    mask_dir = "../people_segmentation/original/masks", pred_mask_dir = "../people_segmentation/original/pred_masks", 
                    threshold = 0.02, save_mask=True):
    image = Image.open(f'{img_dir}/{img_name}.jpg')
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
    upsampled_logits = nn.functional.interpolate(logits,
                    size=image.size[::-1], # (height, width)
                    mode='bilinear',
                    align_corners=False)
    pred_mask = upsampled_logits.argmax(dim=1)[0] == 12
    pred_mask = pred_mask.cpu().detach().numpy().astype(int)
    original_mask = np.asarray(Image.open(f'{mask_dir}/{img_name}.png'))
    
    if show_img==True:
        plt.subplot(1,3,1)
        plt.imshow(original_mask[:,:,0], cmap="gray")
        plt.subplot(1,3,2)
        plt.imshow(pred_mask, cmap = "gray")
        plt.subplot(1,3,3)
        plt.imshow(image)
    if save_mask:
        # save predicted mask
        plt.imsave(f"{pred_mask_dir}/{img_name}.png", pred_mask, cmap='gray')
    return IoU_coeff(original_mask, pred_mask)


feature_extractor_b5 = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
model_b5 = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_b5 = model_b5.to(device)


# orignal files
iou_original = {}

for root, dirs, files in os.walk("../people_segmentation/original/images/", topdown=False):
    for img_name in files:        
        iou_original[img_name] = load_and_get_iou(img_name[:-4], model = model_b5,feature_extractor=feature_extractor_b5, save_mask=True, device=device)

df = pd.DataFrame.from_dict({"name":iou_original.keys(),"iou": iou_original.values()})
df.to_csv("../people_segmentation/original/iou.csv")


for dir_name  in ["0.3", "0.5", "0.7"]:
    for aug_type in ["random_cropped","patch_cropped"]:
        iou_dict = {}
        print("starting ", dir_name, "/", aug_type, " device: ", device )
        for root, dirs, files in os.walk("../people_segmentation/original/images/", topdown=False):
            img_dir = f"../people_segmentation/{dir_name}/{aug_type}/images"
            mask_dir = f"../people_segmentation/{dir_name}/{aug_type}/masks"
            pred_mask_dir = f"../people_segmentation/{dir_name}/{aug_type}/pred_masks"
            for img_name in tqdm(files):
                iou_dict[img_name] = load_and_get_iou(img_name[:-4], model = model_b5,feature_extractor=feature_extractor_b5, 
                                                        device=device, save_mask=True, img_dir=img_dir, mask_dir=mask_dir, 
                                                        pred_mask_dir=pred_mask_dir, show_img=True)

        df = pd.DataFrame.from_dict({"name":iou_dict.keys(),"iou": iou_dict.values()})
        df.to_csv(f"../people_segmentation/{dir_name}/{aug_type}/iou.csv")




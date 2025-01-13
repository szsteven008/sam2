import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device='cpu'))

image = cv2.imread('images/1.jpg')
input_point = None
input_label = None
input_box = np.array([[618, 422, 800, 485], [9, 653,  81, 668]])
#input_box = np.array([[174, 115, 311, 465]]) 1.jpg
#[618, 422, 800, 485], [9, 653,  81, 668] 2.jpg

with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(point_coords=input_point, 
                                         point_labels=input_label, 
                                         box=input_box, 
                                         multimask_output=False)

    masks = np.sum(masks, axis=0, keepdims=True)
    if masks.ndim == 4:
        masks = masks.squeeze(0)
   
    mask = masks[0].astype(np.uint8) * 255

    kernel_size = 9
    mask = cv2.dilate(mask, 
                      np.ones((kernel_size, kernel_size), np.uint8), 
                      iterations=1)
    res_mask = np.zeros(
        (mask.shape[0], mask.shape[1], 4), dtype=np.uint8
    )
    res_mask[mask > 128] = [255, 203, 0, int(255 * 0.73)]
    res_mask = cv2.cvtColor(res_mask, cv2.COLOR_BGRA2RGBA)
    
    res_mask = cv2.threshold(res_mask, 127, 255, cv2.THRESH_BINARY)[1]

    cv2.imwrite('mask.png', res_mask)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image = cv2.addWeighted(image, 1.0, res_mask, 0.3, 0)

    cv2.imshow('image', image)

    cv2.waitKey()
    cv2.destroyAllWindows()

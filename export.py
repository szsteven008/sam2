import argparse
import os
import torch
import cv2
import onnx
import onnxruntime as ort
import numpy as np
from typing import Tuple

from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms

class SAM2ImageEncoder(torch.nn.Module):
    def __init__(
        self,
        sam_model: SAM2Base,
    ) -> None:
        """
        Uses SAM-2 to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam-2): The model to use for mask prediction.
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
            the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
            the maximum area of max_sprinkle_area in low_res_masks.
        """
        super().__init__()
        self.model = sam_model

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    def forward(
        self,
        image: torch.Tensor,
    ):
        backbone_out = self.model.forward_image(image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        image_embed = feats[2]
        high_res_feats1 = feats[0]
        high_res_feats2 = feats[1]

        return image_embed, high_res_feats1, high_res_feats2

def export_encoder(model, output, image):
    onnx_file = output + "/" + "sam2_1.encoder.onnx"

    torch.onnx.export(
        model,
        args = (image,), 
        f = onnx_file,
        input_names = [ "image" ],
        output_names = [ "image_embeddings", "high_res_feats1", "high_res_feats2" ], 
        opset_version = 17, 
        export_params = True, 
        do_constant_folding = True,
        dynamic_axes = {
            "image": { 0: "batch_size", 2: "height", 3: "width" },
            "image_embeddings": { 0: "batch_size", 1: "256", 2: "64", 3: "64"},
            "high_res_feats1": { 0: "batch_size", 1: "32", 2: "256", 3: "256"},
            "high_res_feats2": { 0: "batch_size", 1: "64", 2: "128", 3: "128"},
        },
    )

    print("export sam2_1.encoder.onnx ok!")

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print("check sam2_1.encoder.onnx ok!")

def inference_encoder_onnx(model, output, image):
    onnx_file = output + "/" + "sam2_1.encoder.onnx"
    session = ort.InferenceSession(onnx_file)

    outputs = session.run(None, {
        "image": image.numpy().astype(np.float32)
    })

    image_embed = torch.from_numpy(outputs[0])
    high_res_feats1 = torch.from_numpy(outputs[1])
    high_res_feats2 = torch.from_numpy(outputs[2])

    print(image_embed.shape)
    print(high_res_feats1.shape)
    print(high_res_feats2.shape)

    return image_embed, high_res_feats1, high_res_feats2

class SAM2ImageDecoder(torch.nn.Module):
    def __init__(
        self,
        sam_model: SAM2Base, 
        transforms: SAM2Transforms, 
        mode: str,
    ) -> None:
        """
        Uses SAM-2 to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam-2): The model to use for mask prediction.
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
            the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
            the maximum area of max_sprinkle_area in low_res_masks.
        """
        super().__init__()
        self.model = sam_model
        self.mode = mode
        self._transforms = transforms
        self.mask_threshold = 0.0

    def forward(
        self,
        image_embeddings: torch.Tensor,  # [1,256,64,64]
        high_res_features1: torch.Tensor, # [1, 32, 256, 256]
        high_res_features2: torch.Tensor, # [1, 64, 128, 128]
        boxes: torch.Tensor, # [box_num, 2, 2]
        point_coords: torch.Tensor, # [1, point_num, 2]
        point_labels: torch.Tensor, # [1, point_num]
    ):
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
        """
        if self.mode == "box":
            point_coords = None
            point_labels = None
        elif self.mode == "point":
            boxes = None

        masks, iou_predictions = self._predict(
            image_embeddings, 
            high_res_features1, 
            high_res_features2, 
            boxes, 
            point_coords, 
            point_labels, 
        )

        return masks, iou_predictions

    def _predict(
        self,
        image_embeddings: torch.Tensor,  # [1, 256, 64, 64]
        high_res_features1: torch.Tensor, # [1, 32, 256, 256]
        high_res_features2: torch.Tensor, # [1, 64, 128, 128]
        boxes: torch.Tensor, # [box_num, 2, 2]
        point_coords: torch.Tensor, # [1, point_num, 2]
        point_labels: torch.Tensor, # [1, point_num]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=None,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [ high_res_features1, high_res_features2 ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, self.model.image_size
        )
        masks = masks > self.mask_threshold

        return masks.type(torch.uint8), iou_predictions

def export_decoder(model, 
                   output, 
                   encoder, 
                   mode, 
                   image, 
                   boxes, 
                   point_coords, 
                   point_labels):
    onnx_file = output + "/" + "sam2_1.decoder." + mode + ".onnx"

    (
        image_embed, 
        high_res_feats1, 
        high_res_feats2
    ) = inference_encoder_onnx(encoder, output, image)

    if mode == "box":
        dynamic_axes = {
            "boxes": { 0: "box_num" },
        }
    else:
        dynamic_axes = {
            "point_coords": { 1: "point_num" },
            "point_labels": { 1: "point_num" },
        }

    torch.onnx.export(
        model,
        args = (
            image_embed, 
            high_res_feats1, 
            high_res_feats2, 
            boxes, 
            point_coords, 
            point_labels, 
        ),
        f = onnx_file,
        input_names = [ "image_embeddings", "high_res_features1", "high_res_features2", 
                        "boxes", "point_coords", "point_labels" ],
        output_names = [ "masks", "iou_predictions" ], 
        opset_version = 17, 
        export_params = True, 
        do_constant_folding = True,
        dynamic_axes = dynamic_axes,
    )

    print("export sam2_1.decoder." + mode + ".onnx ok!")

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print("check sam2_1.decoder." + mode + ".onnx ok!")

def inference_decoder_onnx(model, 
                           output, 
                           encoder, 
                           mode, 
                           image, 
                           boxes, 
                           point_coords, 
                           point_labels, 
                           orig_image, 
                           orig_hw):
    (
        image_embed, 
        high_res_feats1, 
        high_res_feats2
    ) = inference_encoder_onnx(encoder, output, image)

    onnx_file = output + "/" + "sam2_1.decoder." + mode + ".onnx"
    session = ort.InferenceSession(onnx_file)

    if mode == "box":
        outputs = session.run(None, {
            "image_embeddings": image_embed.numpy().astype(np.float32), 
            "high_res_features1": high_res_feats1.numpy().astype(np.float32), 
            "high_res_features2": high_res_feats2.numpy().astype(np.float32), 
            "boxes": boxes.numpy().astype(np.int32), 
        })
    elif mode == "point":
        outputs = session.run(None, {
            "image_embeddings": image_embed.numpy().astype(np.float32), 
            "high_res_features1": high_res_feats1.numpy().astype(np.float32), 
            "high_res_features2": high_res_feats2.numpy().astype(np.float32), 
            "point_coords": point_coords.numpy().astype(np.int32), 
            "point_labels": point_labels.numpy().astype(np.int32), 
        })
    else:
        outputs = session.run(None, {
            "image_embeddings": image_embed.numpy().astype(np.float32), 
            "high_res_features1": high_res_feats1.numpy().astype(np.float32), 
            "high_res_features2": high_res_feats2.numpy().astype(np.float32), 
            "boxes": boxes.numpy().astype(np.int32), 
            "point_coords": point_coords.numpy().astype(np.int32), 
            "point_labels": point_labels.numpy().astype(np.int32), 
        })

    masks = np.sum(outputs[0], axis=0, keepdims=True)
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
    res_mask = cv2.resize(res_mask, (orig_hw[1], orig_hw[0]))

    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2BGRA)
    image = cv2.addWeighted(orig_image, 1.0, res_mask, 0.3, 0)

    cv2.imshow("image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def prepare_input_data(transforms: SAM2Transforms, mode: str):
    orig_image = cv2.imread("images/1.jpg")
    orig_hw = orig_image.shape[:2]
    orig_point_coords = np.array([[700, 450]])
    orig_point_labels = np.array([1])

    if mode == "box":
        orig_boxes = np.array([[618, 422, 800, 485], [9, 653,  81, 668]])
    else:
        orig_boxes = np.array([[618, 422, 800, 485]])

    image = transforms(orig_image).unsqueeze(0).to("cpu")

    boxes = transforms.transform_boxes(
        torch.as_tensor(orig_boxes, dtype=torch.float, device="cpu"), 
        normalize=True, 
        orig_hw=orig_hw
    ).type(torch.int)
    point_coords = transforms.transform_coords(
        torch.as_tensor(orig_point_coords, dtype=torch.float, device="cpu"), 
        normalize=True, 
        orig_hw=orig_hw
    ).type(torch.int).unsqueeze(0).to("cpu")
    point_labels = torch.as_tensor(orig_point_labels, dtype=torch.int, device="cpu").unsqueeze(0).to("cpu")

    return orig_image, orig_hw, image, boxes, point_coords, point_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export SAM2 Model to ONNX", add_help=True)
    parser.add_argument("--encode", "-e", help="test encoder.onnx model", action="store_true")
    parser.add_argument("--decode", "-d", help="test decoder.onnx model", action="store_true")
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument(
        "--mode", "-m", type=str, default="box", help="mode: box|point|all"
    )

    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    output_dir = args.output_dir
    mode = args.mode
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)

    model = build_sam2(config_file, checkpoint_path, device="cpu")
    transforms = SAM2Transforms(
        resolution=model.image_size,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
    )

    encoder = SAM2ImageEncoder(model)
    decoder = SAM2ImageDecoder(model, transforms, mode)

    (
        orig_image, 
        orig_hw, 
        image, 
        boxes, 
        point_coords, 
        point_labels
    ) = prepare_input_data(transforms, mode) 

    if args.encode:
        inference_encoder_onnx(model, output_dir, image)
    elif args.decode:
        inference_decoder_onnx(decoder, 
                       output_dir, 
                       encoder, 
                       mode, 
                       image, 
                       boxes, 
                       point_coords, 
                       point_labels, 
                       orig_image, 
                       orig_hw)
    else:
        export_encoder(encoder, output_dir, image)
        export_decoder(decoder, 
                       output_dir, 
                       encoder, 
                       mode, 
                       image, 
                       boxes, 
                       point_coords, 
                       point_labels)
import torch
import torch.nn as nn

import math
import numpy as np
from PIL import Image
import os
from dataclasses import dataclass

from model.net import *
from model.config import RFDETRBaseConfig
from dataset.dataset import get_classes


@dataclass
class EvalArgs:
    conf: float = 0.25
    sahi_fuzz: float = 0.1

    iou: float = 0.0 # UNUSED


def preprocess_chunk(image_paths, target_shape=(1288, 784), grayscale=False):
    """
    Load each image in PIL, then confirm they match the model's input shape.
    """

    batch_imgs = []
    metas = []
    for pindex, p in enumerate(image_paths):
        if isinstance(p, Image.Image):
            image = p.copy()
            p = str(pindex)
        elif isinstance(p, str):
            image = Image.open(p)
        else:
            raise ValueError(f"Unexpected type for image_path: {type(p)}")

        if grayscale:
            image = image.convert("L")
        orig_w, orig_h = image.size
        if orig_w > orig_h:
            new_w = target_shape[0]
            new_h = int((target_shape[1] / orig_w) * orig_h)
        else:
            new_h = target_shape[0]
            new_w = int((target_shape[1] / orig_h) * orig_w)

        resized_img = image.resize((new_w, new_h))
        if grayscale:
            new_image = Image.new("L", target_shape, 0)
        else:
            new_image = Image.new("RGB", target_shape, 0)
        new_image.paste(resized_img, (0,0))

        arr = np.array(new_image)
        if grayscale and arr.ndim==2:
            arr = np.expand_dims(arr, axis=-1)
        if not grayscale:
            arr = arr[..., ::-1].copy()  # rgb->bgr
        batch_imgs.append(arr)
        metas.append((p, orig_w, orig_h, new_w, new_h))
    return np.stack(batch_imgs, axis=0), metas


# Loads a single image into a 4 image SAHI set
def preprocess_image_sahi(image_path, target_shape=(1288, 784), grayscale=False):
    """
    Returns:
        stacked (np.ndarray): Stacked image slices, shape = (4, H, W, C).
        metas (list): List of metadata tuples for each slice.
        Image width and height in pixels.
    """

    # Load image
    if isinstance(image_path, Image.Image):
        image = image_path.copy()
        image_path = str(image_path)
    elif isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        raise ValueError(f"Unexpected type for image_path: {type(image_path)}")

    # Convert as needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode not in ['RGB', 'L']:
        image = image.convert('RGB')
    if grayscale:
        image = image.convert('L')

    orig_w, orig_h = image.size
    if orig_w == 0 or orig_h == 0:
        raise ValueError(f"Image has invalid dimensions: {orig_w}x{orig_h}")

    # Resize to maintain aspect ratio
    if orig_w > orig_h:
        new_w = target_shape[0]
        new_h = int((target_shape[1] / orig_w) * orig_h)
    else:
        new_h = target_shape[0]
        new_w = int((target_shape[1] / orig_h) * orig_w)

    # Create a full resized slice
    full_img = image.resize((new_w, new_h))

    # Pad to a square
    pad_mode = "L" if grayscale else "RGB"
    padded_full = Image.new(pad_mode, target_shape, 0)
    padded_full.paste(full_img, (0, 0))

    metas = [
        # Cx, Cy, Cw, Ch, Rw, Rh, Tw, Th
        ((0, 0, orig_w, orig_h), (new_w, new_h), target_shape), 
    ]

    slice_data = [
        padded_full,
    ]

    # If the image is large enough to slice:
    if orig_w > target_shape[0] and orig_h > target_shape[1]:
        # 20% larger than 1/3 the width
        tile_size = min(target_shape[0], int(orig_w / 3 * 1.2 + 0.5))

        top = (orig_h - tile_size) // 2
        bottom = top + tile_size

        left = 0
        center = (orig_w - tile_size) // 2
        right = orig_w - tile_size

        left_coords = (left, top, left + tile_size, bottom)
        center_coords = (center, top, center + tile_size, bottom)
        right_coords = (right, top, right + tile_size, bottom)

        left_slice = image.crop(left_coords)
        center_slice = image.crop(center_coords)
        right_slice = image.crop(right_coords)

        left_slice = left_slice.resize(target_shape)
        center_slice = center_slice.resize(target_shape)
        right_slice = right_slice.resize(target_shape)

        slice_data.append(left_slice)
        slice_data.append(center_slice) 
        slice_data.append(right_slice)

        metas.append((left_coords, (tile_size, tile_size), (tile_size, tile_size)))
        metas.append((center_coords, (tile_size, tile_size), (tile_size, tile_size)))
        metas.append((right_coords, (tile_size, tile_size), (tile_size, tile_size)))

    slices = []

    for slice_img in slice_data:
        arr = np.array(slice_img)
        # Expand grayscale to (H,W,1)
        if grayscale and arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        # For RGB, ensure BGR format in the output array
        if not grayscale:
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Expected RGB image with 3 channels, got shape {arr.shape}")
            arr = arr[..., ::-1].copy()  # RGB -> BGR
        slices.append(arr)

    stacked = np.stack(slices, axis=0)
    return image_path, stacked, metas, orig_w, orig_h


# Must follow specification:
# https://coda.io/d/Dev-Journal_dM-WYGdU8b5/Model-Input-Output-API_suwoEOJX

class RoboflowDetectorExport(nn.Module):
    def __init__(self, checkpoint_file_path):
        super().__init__()

        try:
            from git import Repo
            repo = Repo('.')
            self.git_hash = repo.head.object.hexsha
        except:
            self.git_hash = "unknown-git-hash"

        self.class_names = get_classes()
        nc = len(self.class_names)

        model_args = RFDETRBaseConfig(pretrain_weights=checkpoint_file_path)
        model_args.num_classes = nc - 1
        self.model_args = model_args

        model = build_model(model_args)
        _, self.postprocessors = build_criterion_and_postprocessors(model_args)

        checkpoint = torch.load(model_args.pretrain_weights, map_location='cpu', weights_only=False)

        # --- Attempt to find the EMA state dict ---
        ema_state_dict = None

        # Expose training metadata from the checkpoint on this object in standard location
        self.train_metadata = checkpoint['saronic_metadata']

        # Expose trained image width/height
        self.width = self.train_metadata['args']['args'].w
        self.height = self.train_metadata['args']['args'].h
        self.channels = 3 # Hardcoded for now

        print(f"train_metadata = {self.train_metadata}")

        # 1. Check if the 'callbacks' key exists
        if 'callbacks' in checkpoint:
            print("\nFound 'callbacks' key. Searching for EMA state inside...")

            # 2. Iterate through the saved callback states
            #    The key for the EMA callback might be 'EMA' or something similar.
            #    We iterate to be sure, checking for the presence of 'ema_state_dict'.
            for callback_key, callback_state in checkpoint['callbacks'].items():
                # Ensure the state is a dictionary and contains our target key
                if isinstance(callback_state, dict) and 'ema_state_dict' in callback_state:
                    print(f"  Found 'ema_state_dict' within callback state keyed by: '{callback_key}'")
                    ema_state_dict = callback_state['ema_state_dict']
                    # You could also optionally check the '_ema_state_dict_ready' flag:
                    if '_ema_state_dict_ready' in callback_state:
                        print(f"  '_ema_state_dict_ready' flag is: {callback_state['_ema_state_dict_ready']}")
                    break # Stop searching once we find it

            if ema_state_dict is not None:
                print(f"\nSuccessfully extracted 'ema_state_dict'!")
                print(f"  Number of tensors/parameters in ema_state_dict: {len(ema_state_dict)}")
                # Now you can use the ema_state_dict, for example, to load into a model:
                # your_model.load_state_dict(ema_state_dict)
            else:
                print("\nCould not find 'ema_state_dict' within any entry under the 'callbacks' key.")
                print("  Possible reasons:")
                print("    - This checkpoint was saved *before* the EMA callback was active or working correctly.")
                print("    - The EMA callback's on_save_checkpoint hook did not run when this checkpoint was saved.")

        else:
            print("\nCheckpoint does not contain the top-level 'callbacks' key.")
            print("  EMA state cannot be retrieved as callback states were not saved in this file.")
            print("  This might be an older checkpoint or saved with different settings.")

        if ema_state_dict is None:
            orig_model_state = checkpoint['model'] if 'model' in checkpoint else checkpoint['state_dict']
        else:
            orig_model_state = ema_state_dict

        self.args_shape = model_args.shape
        self.args_grayscale = False # EO only for now

        # Remove 'model.' prefix from state dict keys
        model_state = {}
        for name, state in orig_model_state.items():
            if name.startswith('model.'):
                model_state[name[6:]] = state
            else:
                model_state[name] = state

        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)

        if len(missing_keys) > 0:
            raise RuntimeError(f"Missing keys in checkpoint: {missing_keys}")
        if len(unexpected_keys) > 1:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected_keys}")

        model.export()
        model.eval()

        #self.model = model.half()
        #self.dtype = torch.float16
        self.model = model
        self.dtype = torch.float32

    def forward(self, images):
        # Preprocessing:
        # Input format is 8-bit BGR images (commonly used OpenCV cv::Mat format) BHWC format.
        # The channel order is BCHW.  The input shape may be square or rectangular.

        # Convert to planar float32 and normalize
        images = images.float().permute(0, 3, 1, 2) / 255.0

        # Apply ImageNet normalization (with BGR channel order)
        imagenet_mean = torch.tensor([0.406, 0.456, 0.485], dtype=torch.float32).view(3, 1, 1)
        imagenet_std = torch.tensor([0.225, 0.224, 0.229], dtype=torch.float32).view(3, 1, 1)
        images.sub_(imagenet_mean).div_(imagenet_std)

        results = self.model(images)

        ltrb = results[0]
        confidences = results[1].sigmoid()

        # Exported model returns a tuple of ([ltrb], [confidences]) tensors.
        # Concat these into a single tensor.
        return torch.cat([ltrb, confidences], dim=2)

    def get_default_eval_args(self):
        return EvalArgs()

    @torch.no_grad()
    def detect(self, image_paths, args=None):
        # If type of args is dict, convert to EvalArgs
        if args is None:
            args = self.get_default_eval_args()
        elif isinstance(args, dict):
            args = EvalArgs(**args)

        batch_array, metas = preprocess_chunk(image_paths, self.args_shape, self.args_grayscale)

        batch_tensor = torch.from_numpy(batch_array).permute(0, 3, 1, 2).float()
        batch_tensor = batch_tensor / 255.0

        # Input is in BGR
        imagenet_mean = torch.tensor([0.406, 0.456, 0.485], dtype=torch.float32).view(3, 1, 1)
        imagenet_std = torch.tensor([0.225, 0.224, 0.229], dtype=torch.float32).view(3, 1, 1)
        batch_tensor.sub_(imagenet_mean).div_(imagenet_std)

        device = next(self.model.parameters()).device
        batch_tensor = batch_tensor.to(device, dtype=self.dtype)

        preds = self.model(batch_tensor)

        # forward_export returns outputs_coord, outputs_class
        preds = {
            "pred_boxes": preds[0],
            "pred_logits": preds[1],
        }

        B = batch_tensor.shape[0] # Batch size
        batch_results = {}

        with torch.no_grad():
            target_sizes_rel = torch.ones(B, 2, dtype=torch.float32, device=device)
            results = self.postprocessors["bbox"](preds, target_sizes_rel)

        for b in range(B):
            (this_path, orig_w, orig_h, new_w, new_h) = metas[b]
            scale_x = new_w / self.args_shape[0]
            scale_y = new_h / self.args_shape[1]

            pred_labels = results[b]["labels"]
            pred_scores = results[b]["scores"]
            pred_boxes_rel = results[b]["boxes"]

            detections = []
            for i in range(pred_labels.shape[0]):
                conf = pred_scores[i].item()
                if not math.isfinite(conf) or conf < args.conf:
                    continue
                c_id = pred_labels[i].item()
                c_name = self.class_names[c_id] if 0 <= c_id < len(self.class_names) else f"class_{c_id}"

                x1, y1, x2, y2 = pred_boxes_rel[i].cpu().tolist()
                x1 /= scale_x
                x2 /= scale_x
                y1 /= scale_y
                y2 /= scale_y

                detections.append({
                    "class_id": c_id,
                    "class_name": c_name,
                    "confidence": conf,
                    "coords_ltrb": [x1, y1, x2, y2],
                })

            batch_results[this_path] = {
                "w": orig_w,
                "h": orig_h,
                "detections": detections,
            }

        return batch_results

    @torch.no_grad()
    def detect_sahi(self, image_paths, args=None):
        # If type of args is dict, convert to EvalArgs
        if args is None:
            args = self.get_default_eval_args()
        elif isinstance(args, dict):
            args = EvalArgs(**args)

        batch_results = {}

        for image_path in image_paths:
            image_tag, batch_array, metas, orig_w, orig_h = preprocess_image_sahi(
                image_path, self.args_shape, self.args_grayscale
            )

            batch_tensor = torch.from_numpy(batch_array).permute(0, 3, 1, 2).float()
            batch_tensor = batch_tensor / 255.0

            # Input is in BGR
            imagenet_mean = torch.tensor([0.406, 0.456, 0.485], dtype=torch.float32).view(3, 1, 1)
            imagenet_std = torch.tensor([0.225, 0.224, 0.229], dtype=torch.float32).view(3, 1, 1)
            batch_tensor.sub_(imagenet_mean).div_(imagenet_std)

            device = next(self.model.parameters()).device
            batch_tensor = batch_tensor.to(device, dtype=self.dtype)

            preds = self.model(batch_tensor)

            # forward_export returns outputs_coord, outputs_class
            preds = {
                "pred_boxes": preds[0],
                "pred_logits": preds[1],
            }

            B = batch_tensor.shape[0] # Batch size

            with torch.no_grad():
                target_sizes_rel = torch.ones(B, 2, dtype=torch.float32, device=device)
                results = self.postprocessors["bbox"](preds, target_sizes_rel)

            detections = []
            full_detections = []
            for b in range(B):
                pred_labels = results[b]["labels"]
                pred_scores = results[b]["scores"]
                pred_boxes_rel = results[b]["boxes"]

                orig_rect, slice_size, tile_shape = metas[b]

                for i in range(pred_labels.shape[0]):
                    conf = pred_scores[i].item()
                    if not math.isfinite(conf) or conf < args.conf:
                        continue
                    c_id = pred_labels[i].item()
                    c_name = self.class_names[c_id] if 0 <= c_id < len(self.class_names) else f"class_{c_id}"

                    ltrb = pred_boxes_rel[i].cpu()

                    # Convert to fraction within the slice
                    ltrb[0] = ltrb[0] * (tile_shape[0] / slice_size[0])
                    ltrb[1] = ltrb[1] * (tile_shape[1] / slice_size[1])
                    ltrb[2] = ltrb[2] * (tile_shape[0] / slice_size[0])
                    ltrb[3] = ltrb[3] * (tile_shape[1] / slice_size[1])

                    # Convert to original image coords
                    orig_crop_w = orig_rect[2] - orig_rect[0]
                    orig_crop_h = orig_rect[3] - orig_rect[1]
                    ltrb[0] = (orig_rect[0] + ltrb[0] * orig_crop_w) / orig_w
                    ltrb[1] = (orig_rect[1] + ltrb[1] * orig_crop_h) / orig_h
                    ltrb[2] = (orig_rect[0] + ltrb[2] * orig_crop_w) / orig_w
                    ltrb[3] = (orig_rect[1] + ltrb[3] * orig_crop_h) / orig_h

                    if i == 0:
                        full_detections.append((ltrb, conf, c_id))
                    else:
                        detections.append((ltrb, conf, c_id))

            def is_contained(detection, full_detection, fuzz_factor=0.1):
                det_box, det_conf, det_id = detection
                full_box, full_conf, full_id = full_detection
                # Only consider containment if full_detection has higher confidence
                if det_id != full_id:
                    return False
                if full_conf <= det_conf:
                    return False
                # Fuzz margin
                width = full_box[2] - full_box[0]
                height = full_box[3] - full_box[1]
                margin_x = width * fuzz_factor
                margin_y = height * fuzz_factor
                # Check bounding
                is_contained_x = (det_box[0] >= full_box[0] - margin_x and
                                  det_box[2] <= full_box[2] + margin_x)
                is_contained_y = (det_box[1] >= full_box[1] - margin_y and
                                  det_box[3] <= full_box[3] + margin_y)
                return is_contained_x and is_contained_y

            filtered_detections = []
            for detection in detections:
                is_contained_in_any = False
                for full_detection in full_detections:
                    if is_contained(detection, full_detection, args.sahi_fuzz):
                        is_contained_in_any = True
                        break
                if not is_contained_in_any:
                    filtered_detections.append(detection)

            detections = filtered_detections + full_detections

            # Convert format (a bit crufty)
            final_dets = []
            if detections:
                boxes_np = np.array([d[0] for d in detections])
                confs_np = np.array([d[1] for d in detections])
                cls_ids  = np.array([d[2] for d in detections])

                for i in range(boxes_np.shape[0]):
                    final_dets.append({
                        "class_id": int(cls_ids[i]),
                        "confidence": float(confs_np[i]),
                        "coords_ltrb": boxes_np[i].tolist()
                    })

            result_dets = []
            for dd in final_dets:
                ltrb = dd["coords_ltrb"]
                c_id = dd["class_id"]
                c_name = self.class_names[c_id] if 0 <= c_id < len(self.class_names) else f"class_{c_id}"
                result_dets.append({
                    "class_id": int(c_id),
                    "class_name": c_name,
                    "confidence": dd["confidence"],
                    "coords_ltrb": ltrb,
                })

            batch_results[image_tag] = {
                "w": orig_w,
                "h": orig_h,
                "detections": result_dets,
            }

        return batch_results

#!/usr/bin/env python3
"""
SAM2 Part Segmentation CLI Tool
Segments manufacturing parts from images using SAM2 tiny model.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def load_model(checkpoint_path: str, model_cfg: str = "sam2_hiera_t.yaml") -> SAM2AutomaticMaskGenerator:
    """
    Load SAM2 tiny model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint file
        model_cfg: Model configuration name

    Returns:
        SAM2AutomaticMaskGenerator instance
    """
    print(f"Loading SAM2 model from {checkpoint_path}...")

    # Build SAM2 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sam2 = build_sam2(
        model_cfg,
        checkpoint_path,
        device=device,
        apply_postprocessing=False
    )

    # Create automatic mask generator
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.60,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    print("Model loaded successfully!")
    return mask_generator


def segment_image(image_path: str, mask_generator: SAM2AutomaticMaskGenerator) -> dict:
    """
    Segment objects in an image using automatic mask generation.

    Args:
        image_path: Path to input image
        mask_generator: Initialized SAM2AutomaticMaskGenerator

    Returns:
        Dictionary containing largest mask data
    """
    print(f"\nProcessing image: {image_path}")

    # Read image with OpenCV and convert BGR to RGB
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print(f"Image shape: {image_rgb.shape}")
    print("Generating masks (this may take ~10-60 seconds)...")

    # Generate masks
    masks = mask_generator.generate(image_rgb)

    if not masks:
        raise ValueError("No masks generated from image")

    print(f"Generated {len(masks)} masks")

    # Filter out background by detecting edge-touching masks
    # Background often touches image borders
    def is_likely_background(mask_data, img_shape, margin=10):
        x, y, w, h = mask_data['bbox']
        img_height, img_width = img_shape[:2]

        # Check if bbox touches any edge
        touches_edge = (
            x < margin or
            y < margin or
            x + w > img_width - margin or
            y + h > img_height - margin
        )

        # Also check if mask area is very large (>60% of image)
        image_area = img_height * img_width
        is_very_large = mask_data['area'] > 0.6 * image_area

        return touches_edge and is_very_large

    # Filter out likely background masks
    non_background_masks = [
        m for m in masks
        if not is_likely_background(m, image_rgb.shape)
    ]

    print(f"Filtered out {len(masks) - len(non_background_masks)} likely background masks")
    print(f"Remaining masks: {len(non_background_masks)}")

    # Sort by area and select top 3
    masks_by_area = sorted(non_background_masks, key=lambda x: x['area'], reverse=True)
    selected_masks = masks_by_area[:3] if masks_by_area else []

    print(f"\nSelected top {len(selected_masks)} object(s) by area:")
    for i, mask in enumerate(selected_masks, 1):
        print(f"  Object {i}:")
        print(f"    Area: {mask['area']} pixels")
        print(f"    Bounding box (x, y, w, h): {mask['bbox']}")
        print(f"    Predicted IoU: {mask['predicted_iou']:.3f}")
        print(f"    Stability score: {mask['stability_score']:.3f}")

    return {
        'masks': selected_masks,
        'original_image': image_bgr,
        'original_image_rgb': image_rgb
    }


def draw_crosshair_and_label(image: np.ndarray, mask: np.ndarray, bbox: list,
                             label: str = "object", min_size: int = 50) -> np.ndarray:
    """
    Draw crosshair at mask centroid with optional label.

    Args:
        image: Image to draw on (BGR format)
        mask: Binary segmentation mask (2D boolean/uint8 array)
        bbox: Bounding box [x, y, width, height]
        label: Text label to display
        min_size: Minimum bbox size to show label

    Returns:
        Image with crosshair and label drawn
    """
    x, y, w, h = [int(v) for v in bbox]

    # Calculate centroid (center of mass) from mask
    # Moments are statistical measures of the shape
    mask_uint8 = mask.astype(np.uint8) * 255
    M = cv2.moments(mask_uint8)

    # m00 = total area (sum of all white pixels)
    # m10 = sum of x-coordinates weighted by pixel intensity
    # m01 = sum of y-coordinates weighted by pixel intensity
    # Centroid = (m10/m00, m01/m00) = weighted average position
    if M['m00'] > 0:
        center_x = int(M['m10'] / M['m00'])
        center_y = int(M['m01'] / M['m00'])
    else:
        # Fallback to bbox center if moments fail
        center_x = x + w // 2
        center_y = y + h // 2

    # Draw crosshair (white with black outline for visibility)
    crosshair_size = 20
    thickness = 2

    # Horizontal line
    cv2.line(image, (center_x - crosshair_size, center_y),
             (center_x + crosshair_size, center_y), (0, 0, 0), thickness + 2)
    cv2.line(image, (center_x - crosshair_size, center_y),
             (center_x + crosshair_size, center_y), (255, 255, 255), thickness)

    # Vertical line
    cv2.line(image, (center_x, center_y - crosshair_size),
             (center_x, center_y + crosshair_size), (0, 0, 0), thickness + 2)
    cv2.line(image, (center_x, center_y - crosshair_size),
             (center_x, center_y + crosshair_size), (255, 255, 255), thickness)

    # Draw center dot
    cv2.circle(image, (center_x, center_y), 4, (0, 0, 0), -1)
    cv2.circle(image, (center_x, center_y), 3, (255, 255, 255), -1)

    # Draw label if bbox is large enough
    if w >= min_size and h >= min_size:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Position text below center
        text_x = center_x - text_w // 2
        text_y = center_y + crosshair_size + text_h + 10

        # Draw background rectangle
        padding = 5
        cv2.rectangle(image,
                     (text_x - padding, text_y - text_h - padding),
                     (text_x + text_w + padding, text_y + baseline + padding),
                     (0, 0, 0), -1)

        # Draw text
        cv2.putText(image, label, (text_x, text_y), font, font_scale,
                   (255, 255, 255), font_thickness)

    return image


def save_outputs(result: dict, output_dir: str = "output"):
    """
    Save binary masks and overlay visualization for multiple objects.

    Args:
        result: Dictionary containing segmentation results
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)

    masks_data = result['masks']
    original_bgr = result['original_image']

    # Save individual binary masks
    for i, mask_data in enumerate(masks_data, 1):
        mask = mask_data['segmentation']
        mask_binary = (mask * 255).astype(np.uint8)
        mask_path = os.path.join(output_dir, f"mask_{i}.png")
        cv2.imwrite(mask_path, mask_binary)
        print(f"\nSaved binary mask {i} to: {mask_path}")

    # Create green overlay with all objects
    overlay = original_bgr.copy()
    green_mask = np.zeros_like(original_bgr)
    green_mask[:, :] = [0, 255, 0]  # Green in BGR

    # Apply all masks with green overlay
    for i, mask_data in enumerate(masks_data):
        mask = mask_data['segmentation']
        bbox = mask_data['bbox']

        # Apply mask with transparency
        mask_3channel = np.stack([mask] * 3, axis=-1)
        overlay = np.where(mask_3channel,
                          cv2.addWeighted(overlay, 1.0, green_mask, 0.4, 0),
                          overlay)

        # Draw crosshair and label for each object
        overlay = draw_crosshair_and_label(overlay, mask, bbox, label=f"object_{i+1}")

    overlay_path = os.path.join(output_dir, "overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"\nSaved overlay visualization to: {overlay_path}")

    # Print summary metadata
    print(f"\nSegmentation Summary:")
    print(f"  Total objects detected: {len(masks_data)}")
    for i, mask_data in enumerate(masks_data, 1):
        print(f"  Object {i}:")
        print(f"    Bounding box (x, y, w, h): {mask_data['bbox']}")
        print(f"    Area: {mask_data['area']} pixels")
        print(f"    Confidence (IoU): {mask_data['predicted_iou']:.3f}")
        print(f"    Stability score: {mask_data['stability_score']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Segment manufacturing parts from images using SAM2"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to input image file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sam2_hiera_tiny.pt",
        help="Path to SAM2 checkpoint file (default: checkpoints/sam2_hiera_tiny.pt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for masks and overlays (default: output)"
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="sam2_hiera_t.yaml",
        help="Model configuration name (default: sam2_hiera_t.yaml)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}", file=sys.stderr)
        print("\nPlease download the SAM2 tiny checkpoint:", file=sys.stderr)
        print("  mkdir -p checkpoints", file=sys.stderr)
        print("  wget -O checkpoints/sam2_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt", file=sys.stderr)
        sys.exit(1)

    try:
        # Load model
        mask_generator = load_model(args.checkpoint, args.model_cfg)

        # Segment image
        result = segment_image(args.image_path, mask_generator)

        # Save outputs
        save_outputs(result, args.output_dir)

        print("\nSegmentation completed successfully!")

    except Exception as e:
        print(f"\nError during segmentation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

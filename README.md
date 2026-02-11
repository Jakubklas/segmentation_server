# SAM2 Part Segmentation Tool

A standalone Python CLI tool that segments manufacturing parts from images using Meta's Segment Anything Model 2 (SAM2) with the tiny model variant optimized for CPU usage.

## Features

- Automatic object segmentation (no manual prompts required)
- CPU-compatible SAM2 tiny model
- Binary mask generation
- Visual overlay output with green highlighting
- Metadata reporting (bounding box, area, confidence scores)

## Project Structure

```
visual_inspector/
├── segment.py           # Main CLI script
├── req.txt             # Python dependencies
├── README.md           # This file
├── venv/               # Virtual environment
├── checkpoints/        # Model checkpoint storage
│   └── sam2_hiera_tiny.pt
├── input/              # Test images directory
└── output/             # Generated masks and overlays
    ├── mask.png        # Binary segmentation mask
    └── overlay.png     # Green overlay on original image
```

## Setup

### 1. Virtual Environment

The virtual environment is already created. Activate it:

```bash
source venv/bin/activate
```

### 2. Install Dependencies

Dependencies are already installed. If you need to reinstall:

```bash
pip install -r req.txt
```

### 3. Download SAM2 Model Checkpoint

Download the SAM2 tiny model checkpoint:

```bash
mkdir -p checkpoints
wget -O checkpoints/sam2_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
```

Alternative download link if the above doesn't work:
```bash
curl -L -o checkpoints/sam2_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
```

The checkpoint file is approximately 150MB.

## Usage

### Basic Usage

```bash
python segment.py path/to/your/image.jpg
```

### With Custom Options

```bash
python segment.py input/part.jpg \
  --checkpoint checkpoints/sam2_hiera_tiny.pt \
  --output-dir output \
  --model-cfg sam2_hiera_t.yaml
```

### Command Line Arguments

- `image_path` (required): Path to input image file
- `--checkpoint`: Path to SAM2 checkpoint file (default: `checkpoints/sam2_hiera_tiny.pt`)
- `--output-dir`: Output directory for results (default: `output`)
- `--model-cfg`: Model configuration name (default: `sam2_hiera_t.yaml`)

### Example

```bash
# Place your test image in the input directory
cp ~/Downloads/part_image.jpg input/

# Run segmentation
python segment.py input/part_image.jpg

# Check outputs
ls output/
# mask.png - binary segmentation mask
# overlay.png - visual overlay with green highlight
```

## Output Files

### mask.png
Binary segmentation mask where:
- White (255) = segmented object
- Black (0) = background

### overlay.png
Original image with green overlay blended on the segmented region for easy visualization.

### Console Output

The script prints:
- Model loading status
- Image processing progress
- Number of masks generated
- Selected mask metadata:
  - Bounding box coordinates (x, y, width, height)
  - Area in pixels
  - Confidence score (predicted IoU)
  - Stability score

Example output:
```
Loading SAM2 model from checkpoints/sam2_hiera_tiny.pt...
Using device: cpu
Model loaded successfully!

Processing image: input/part.jpg
Image shape: (1080, 1920, 3)
Generating masks (this may take ~10-60 seconds)...
Generated 15 masks

Selected largest mask:
  Area: 245678 pixels
  Bounding box (x, y, w, h): [320, 150, 800, 600]
  Predicted IoU: 0.892
  Stability score: 0.945

Saved binary mask to: output/mask.png
Saved overlay visualization to: output/overlay.png

Metadata:
  Bounding box (x, y, width, height): [320, 150, 800, 600]
  Area: 245678 pixels
  Confidence (IoU): 0.892
  Stability score: 0.945

Segmentation completed successfully!
```

## Performance

- Initial model load: ~10 seconds (CPU)
- Segmentation per image: ~30-60 seconds (CPU)
- Memory usage: ~2-4GB RAM

For faster processing, use a GPU-enabled system where the script will automatically detect and use CUDA.

## Technical Details

### Model Configuration

The script uses SAM2 tiny model with these parameters:
- `points_per_side`: 32 (sampling density)
- `pred_iou_thresh`: 0.7 (quality threshold)
- `stability_score_thresh`: 0.85 (consistency threshold)
- `min_mask_region_area`: 100 pixels (noise filter)

### Segmentation Logic

1. Loads image and converts BGR to RGB
2. Runs automatic mask generation (finds all objects)
3. Sorts masks by area
4. Selects largest mask (assumed to be main part)
5. Generates binary mask and visual overlay

## Troubleshooting

### Model checkpoint not found
```
Error: Checkpoint file not found: checkpoints/sam2_hiera_tiny.pt
```
Download the checkpoint using the command in Setup section.

### Image loading fails
```
Error: Failed to load image: path/to/image.jpg
```
Ensure the image path is correct and the file is a valid image format (JPG, PNG, etc.).

### No masks generated
```
Error: No masks generated from image
```
Try adjusting the mask generator parameters in the `load_model()` function or ensure the image contains visible objects.

### Out of memory
If you encounter memory errors, try:
- Using a smaller input image
- Reducing `points_per_side` parameter
- Closing other applications

## Future Enhancements

- Point-prompt mode for user-guided segmentation
- Multi-object selection and individual mask export
- Batch processing for multiple images
- GPU optimization flags
- Interactive visualization tool
- Configuration file support

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- SAM2 (Meta)
- OpenCV
- NumPy
- Pillow

See [req.txt](req.txt) for complete list.

## License

This tool uses Meta's SAM2 model. Please refer to the [SAM2 license](https://github.com/facebookresearch/segment-anything-2) for usage terms.
# segmentation_server

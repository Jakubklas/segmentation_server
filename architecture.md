# Manufacturing Part Segmentation System

## Overview
Web-based image segmentation tool using SAM2 for manufacturing parts. Upload images from phone → segment automatically → view results.

## Architecture

### Components

**1. Python SAM Microservice** (Port 5000)
- Framework: FastAPI
- Model: SAM2 (tiny variant)
- Purpose: Perform segmentation inference

**2. Axum Web Server** (Port 3000)
- Framework: Rust + Axum + HTMX
- Purpose: Handle uploads, proxy to Python, serve UI

**3. ngrok Tunnel**
- Expose localhost:3000 to phone over HTTPS

## System Diagram
```
Phone (Browser)
    ↓ HTTPS
ngrok (xyz.ngrok.io)
    ↓
Axum Server (:3000)
    ↓ HTTP
Python SAM Service (:5000)
    ↓
File System (/tmp/uploads, /tmp/results)
```

## File Structure
```
manufacturing-inspector/
├── python_service/
│   ├── main.py              # FastAPI app
│   ├── segment.py           # SAM2 logic
│   ├── requirements.txt
│   └── checkpoints/
│       └── sam2_hiera_tiny.pt
│
├── axum_server/
│   ├── src/
│   │   ├── main.rs          # Routes & server
│   │   ├── handlers.rs      # Upload/segment handlers
│   │   └── sam_client.rs    # HTTP client to Python
│   ├── templates/
│   │   └── index.html       # HTMX UI
│   ├── Cargo.toml
│   └── static/              # CSS/images
│
└── tmp/
    ├── uploads/             # Original images
    └── results/             # Segmented masks
```

## API Specification

### Python Service (FastAPI)

**POST /segment**
```json
Request:
{
  "image_path": "/tmp/uploads/abc123.jpg",
  "mode": "automatic"  // or "point" with coords
}

Response:
{
  "mask_path": "/tmp/results/abc123_mask.png",
  "masks": [
    {
      "bbox": [x, y, w, h],
      "area": 12345,
      "confidence": 0.95
    }
  ]
}
```

**GET /health**
```json
Response:
{
  "status": "ok",
  "model_loaded": true
}
```

### Axum Server

**GET /**
- Serves HTML upload form

**POST /upload**
- Multipart form data
- Saves image to /tmp/uploads/
- Returns image ID

**POST /segment**
- Calls Python service
- Returns HTML with results

**GET /static/:file**
- Serves images/CSS

## Data Flow

1. Phone uploads image → Axum saves to `/tmp/uploads/{uuid}.jpg`
2. Axum → POST to Python `/segment` with image path
3. Python loads image, runs SAM2, saves mask to `/tmp/results/{uuid}_mask.png`
4. Python returns mask path + metadata
5. Axum renders result page with original + mask overlay
6. Phone displays results with download option

## Startup Procedure
```bash
# Terminal 1: Python service
cd python_service
pip install -r requirements.txt
python main.py
# Loads SAM2 model (~10s), listens on :5000

# Terminal 2: Axum server
cd axum_server
cargo run
# Listens on :3000

# Terminal 3: ngrok
ngrok http 3000
# Copy HTTPS URL to phone browser
```

## Tech Decisions

- **Separate Python service**: Model persistence (avoid 10s reload per request)
- **File-based exchange**: Simpler than base64 encoding large images
- **HTMX over React**: Lighter weight, better mobile performance
- **Automatic mode first**: No point-selection UI (MVP), add later
- **Tiny model**: Acceptable quality-speed tradeoff for CPU inference

## Phase 1 MVP Checklist

- [ ] Python SAM service operational
- [ ] Axum upload + proxy working
- [ ] HTMX form mobile-optimized
- [ ] ngrok tunnel configured
- [ ] End-to-end flow tested: upload → segment → view

## Future Enhancements

- Point-prompt mode (click to segment specific object)
- Video support (SAM2 tracking)
- Multiple mask visualization
- Defect detection on segmented regions
- Dimension measurement with calibration
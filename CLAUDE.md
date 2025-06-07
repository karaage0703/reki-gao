# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**reki-gao** is a web application that finds historical figures from the ROIS-CODH "KaoKore" dataset that resemble modern face photos. It uses computer vision for face detection and encoding, combined with vector similarity search to match faces across time periods.

### Core Architecture

The application consists of several key components:

1. **Face Detection Pipeline** (`src/face_detection.py`): OpenCV Haar Cascade-based face detection and preprocessing
2. **Face Encoding** (`src/face_encoding.py`): Extracts 128-dimensional feature vectors from detected faces
3. **KaoKore Integration** (`src/kaokore_loader.py`): Manages the historical portrait dataset
4. **Similarity Search** (`src/kaokore_similarity_search.py`): scikit-learn NearestNeighbors for vector similarity matching
5. **FastAPI Web Server** (`src/api.py`): REST API with Web GUI integration

### Data Flow

1. User uploads modern face photo → Face detection → Feature extraction
2. Compare against pre-processed KaoKore dataset vectors
3. Return top-k similar historical portraits with metadata

## Development Commands

### Setup and Installation
```bash
# Initial setup (recommended)
./scripts/setup.sh

# Manual setup
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt

# Download KaoKore dataset
cd data/kaokore
git clone https://github.com/rois-codh/kaokore.git
cd kaokore
cp ../../../scripts/download_kaokore.py ./download.py
python download.py
```

### Running the Application
```bash
# Start server (using start script)
./scripts/start.sh

# Start server (manual)
python -m src.main

# Development mode with auto-reload
python -m src.main --host 0.0.0.0 --port 8000
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Control dataset size for faster startup
python -m src.main --max-images 100    # Process only 100 images
python -m src.main --max-images 0      # Process all images (slow)
```

### Testing and Code Quality
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_face_detection.py

# Code formatting and linting
ruff check .
ruff format .
```

## Configuration

### Environment Variables
Key settings in `.env` file:
- `KAOKORE_MAX_IMAGES`: Limits processed images for faster startup (default: 50)
- `FACE_CONFIDENCE_THRESHOLD`: Face detection sensitivity (default: 0.8)
- `SIMILARITY_THRESHOLD`: Minimum similarity for search results (default: 0.6)
- `DEBUG`: Enable debug mode (default: false)

### Performance Tuning
- **Startup Speed**: Set `KAOKORE_MAX_IMAGES=100` for development (2-3 second startup vs minutes for full dataset)
- **Memory Usage**: Full dataset (~7,500 images) uses 1GB+ memory; limited dataset uses ~100MB
- **API Endpoints**: Access WebGUI at `http://localhost:8000/` and API docs at `http://localhost:8000/docs`

## Development Notes

### Working with Face Detection
- Face detection uses OpenCV Haar Cascades (not deep learning models)
- Images are preprocessed to 128x128 for feature extraction
- Multiple faces in an image are supported but only the first is used for search

### KaoKore Dataset Integration
- Historical portraits are from ancient Japanese artwork
- Metadata includes character names, eras, tags, and artwork sources
- Images are pre-processed into 128-dimensional feature vectors at startup
- Dataset can be limited via configuration for development purposes

### API Design
- RESTful endpoints under `/api/v1/`
- File upload limits: 10MB max, JPEG/PNG only
- Similarity search returns top-k results with confidence scores
- Metadata endpoints provide detailed information about historical figures
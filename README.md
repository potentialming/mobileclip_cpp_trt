# MobileClip TensorRT Classifier

A C++ implementation of MobileClip image classification using TensorRT for high-performance inference.

## Features

- TensorRT 10.x engine optimization with FP16 support
- Batch processing capability for both image and text encoders
- BPE tokenizer for text preprocessing
- Helper scripts for model download and ONNX export

## Project Structure

```
mobileclip_cpp_trt/
├── CMakeLists.txt           # CMake build configuration
├── README.md                # This file
├── config/
│   └── config.yaml          # Configuration file (not used yet)
├── include/                 # Header files (.hpp)
│   ├── classifier.hpp
│   ├── image_encoder_trt.hpp
│   ├── image_preprocess.hpp
│   ├── logger.hpp
│   ├── text_encoder_trt.hpp
│   ├── text_preprocess.hpp
│   └── tokenizer.hpp
├── src/                     # Source files (.cpp)
│   ├── classifier.cpp
│   ├── image_encoder_trt.cpp
│   ├── image_preprocess.cpp
│   ├── logger.cpp
│   ├── main.cpp             # Entry point
│   ├── text_encoder_trt.cpp
│   ├── text_preprocess.cpp
│   └── tokenizer.cpp
├── scripts/                 # Helper Python scripts
│   ├── download_mobileclip_models.py  # Download pretrained models
│   └── export_mobileclip_onnx.py      # Export to ONNX format
├── models/                  # Model files
│   ├── image_encoder.onnx
│   ├── text_encoder.onnx
│   ├── vocab.json
│   ├── merges.txt
│   ├── *.engine             # Auto-generated TRT engines
├── images/                  # Test images
└── build/                   # Build directory
    └── classifier           # Compiled executable
```

## Requirements

### C++ Dependencies
- C++ Compiler: GCC 9.5+ with C++17 support
- CMake: 3.14+
- CUDA: 12.0+
- TensorRT: 10.x
- OpenCV: 4.6.0+

### Python Dependencies (for scripts)
- Python 3.8+
- PyTorch
- open_clip
- mobileclip
- huggingface_hub

Install Python packages:
```bash
pip install torch open_clip_torch huggingface_hub
pip install git+https://github.com/apple/ml-mobileclip.git
```

## Quick Start

### Step 1: Download Pretrained Models

Use the provided script to download MobileCLIP models from Hugging Face:

```bash
# Download MobileCLIP2-S0 (recommended)
python scripts/download_mobileclip_models.py --model mobileclip2

# Or download MobileCLIP-S2
python scripts/download_mobileclip_models.py --model mobileclip

# Custom cache directory
python scripts/download_mobileclip_models.py --model mobileclip2 --cache_dir /path/to/cache
```

Models will be saved to `models_cache/` by default.

### Step 2: Export to ONNX

Convert PyTorch models to ONNX format:

```bash
# Export MobileCLIP2-S0 to ONNX
python scripts/export_mobileclip_onnx.py \
    --model_name MobileCLIP2-S0 \
    --pretrained models_cache/mobileclip2/MOBILECLIP2-S0/mobileclip2_s0.pt \
    --save_dir models \
    --image_res 224 \
    --opset 17

# For FP16 weights (optional)
python scripts/export_mobileclip_onnx.py \
    --model_name MobileCLIP2-S0 \
    --pretrained models_cache/mobileclip2/MOBILECLIP2-S0/mobileclip2_s0.pt \
    --save_dir models \
    --fp16
```

This will generate:
- `models/image_encoder.onnx` - Image encoder
- `models/text_encoder.onnx` - Text encoder
- `models/preprocess.txt` - Preprocessing configuration

You also need to copy tokenizer files:
```bash
cp models_cache/mobileclip2/MOBILECLIP2-S0/vocab.json models/
cp models_cache/mobileclip2/MOBILECLIP2-S0/merges.txt models/
```

### Step 3: Build C++ Project

```bash
mkdir -p build
cd build
cmake ..
make -j8
```

### Step 4: Run Inference

```bash
cd ..
./build/classifier config/config.yaml
```

TensorRT engines will be built automatically on first run.

## Usage

Edit `src/main.cpp` to change test image and labels:

```cpp
config.test_image = "images/your_image.jpg";
config.text_labels = {
    "label 1",
    "label 2"
};
```

Then rebuild and run.

## Python Scripts Reference

### download_mobileclip_models.py

Download pretrained MobileCLIP/MobileCLIP2 models from Hugging Face.

**Options:**
- `--model`: Model type (`mobileclip` or `mobileclip2`, default: `mobileclip2`)
- `--cache_dir`: Cache directory path (default: `models_cache`)

**Available Models:**
- MobileCLIP: S0, S1, S2, B (LT variants available)
- MobileCLIP2: S0, S1, S2, B, S3, S4

### export_mobileclip_onnx.py

Export MobileCLIP/MobileCLIP2 PyTorch models to ONNX format.

**Options:**
- `--model_name`: Model name (e.g., `MobileCLIP2-S0`)
- `--pretrained`: Path to .pt checkpoint file (required)
- `--save_dir`: Output directory (default: `models`)
- `--image_res`: Image resolution, 224 or 336 (default: 224)
- `--opset`: ONNX opset version (default: 17)
- `--fp16`: Export weights as FP16 (optional)

**Note:** Models S0/S2/B use zero normalization (mean=0, std=1), while S3/S4/L-14 use CLIP normalization.

## Example Output

```
=== Similarity Matrix ===
  [0] the red light is on: 0.3281
  [1] the yellow light is on: 0.2595
  [2] the green light is on: 0.2745
  [3] no light is on: 0.2288

=== Prediction ===
Best match: the red light is on
Confidence: 0.3281
```

## Architecture

- **ImageEncoderTRT**: Image to 512-dim embedding
- **TextEncoderTRT**: Text to 512-dim embedding (batch support)
- **Tokenizer**: BPE tokenizer for CLIP text
- **Classifier**: Cosine similarity computation

## Performance

- First run: 1-5 min (TensorRT engine building)
- Subsequent runs: ~5-10ms per image-text pair

## License

Same as MobileClip

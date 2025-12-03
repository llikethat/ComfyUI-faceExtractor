# Face Extractor for ComfyUI

**Version 4.0 - Generic Face Extraction Tool**

Memory-efficient face extraction with reference-based matching. Works with videos of any length.

---

## ⚠️ IMPORTANT: License Information

This tool supports multiple face detection backends with **different licenses**:

| Backend | License | Commercial Use | Quality | GPU |
|---------|---------|----------------|---------|-----|
| **insightface** | Non-Commercial | ❌ NO | ⭐⭐⭐⭐⭐ Best | ✅ |
| **mediapipe** | Apache 2.0 | ✅ YES | ⭐⭐⭐⭐ Good | ❌ |
| **opencv_cascade** | BSD | ✅ YES | ⭐⭐⭐ Basic | ❌ |

### For Commercial Projects

**DO NOT use `insightface` backend for commercial work!**

Use `mediapipe` or `opencv_cascade` instead:

```
detection_backend: "mediapipe"   # Recommended for commercial
```

### InsightFace License Notice

InsightFace is provided under a **non-commercial license**:

> The InsightFace project is released under the MIT License for non-commercial purposes only.
> For commercial use, please contact the authors for licensing terms.
> 
> Source: https://github.com/deepinsight/insightface#license

If you need InsightFace quality for commercial work, contact InsightFace for a commercial license.

---

## Features

- **Built-in Video Loading**: No VHS dependency needed
- **Memory-Efficient**: Streaming mode uses ~500MB regardless of video length
- **Aspect Ratio Preservation**: No more squashed faces
- **GPU Memory Management**: Automatic flush after each chunk
- **Multiple Backends**: Choose based on quality vs license needs
- **Browser-Friendly**: Minimal data sent back to browser

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/llikethat/ComfyUI_faceExtractor.git
cd comfyui_face_extractor

# For non-commercial use (best quality):
pip install insightface onnxruntime-gpu mediapipe psutil

# For commercial use (good quality):
pip install mediapipe psutil
```

---

## Quick Start

1. Load reference image(s) of target face
2. Set `video_path` to your video file (NOT VHS_LoadVideo!)
3. Choose `detection_backend` based on your license needs
4. Run - faces saved to `ComfyUI/output/face_extract_001/`

---

## Nodes

### Face Reference Embedding

Creates face embedding from reference images.

| Input | Description |
|-------|-------------|
| reference_images | One or more face images |
| detection_backend | "insightface", "mediapipe", or "opencv_cascade" |
| min_confidence | Minimum detection confidence (0.5 default) |

### Face Extractor

Main extraction node with built-in video loading.

| Input | Description |
|-------|-------------|
| **video_path** | Direct path to video (RECOMMENDED) |
| images | Alternative: IMAGE from other nodes |
| reference_embedding | From Face Reference Embedding |
| similarity_threshold | Match threshold (0.6 default) |
| margin_factor | Padding around face (0.4 default) |
| output_size | Output resolution, aspect preserved (512 default) |
| max_faces_per_frame | Max faces per frame (1 default) |
| frame_skip | Process every Nth frame (1 default) |
| processing_mode | "streaming" or "chunked" |
| memory_threshold_percent | RAM limit for chunked mode (75% default) |
| **chunk_size** | **Frames per chunk (0 = auto-calculate)** |
| detection_backend | Choose based on license needs |

### Face Matcher

Compare two faces for threshold tuning.

---

## Processing Modes

### Streaming (Default) ⭐
- One frame at a time
- ~500 MB RAM constant
- Best for: Large videos, limited RAM

### Chunked
- Batches of frames processed together
- GPU flushed after EVERY chunk
- Faster than streaming
- Best for: When you have RAM to spare

**Chunk Size Configuration:**
- `chunk_size = 0` (default): Auto-calculate based on available memory and `memory_threshold_percent`
- `chunk_size = 100`: Process exactly 100 frames per chunk
- `chunk_size = 500`: Process 500 frames per chunk (faster, uses more RAM)

Console output shows which mode is used:
```
[Face Extractor] Auto chunk size: 287 (based on 75.0% memory threshold)
# or
[Face Extractor] Using manual chunk size: 500
```

---

## Aspect Ratio Preservation

**Before (v3)**: Faces were squashed to square
```
Original: 400x500 → Forced to 512x512 → Distorted!
```

**Now (v4)**: Aspect ratio preserved with padding
```
Original: 400x500 → Scaled to 409x512 → Centered on 512x512 canvas
```

---

## Memory Optimization

### GPU Memory
- Flushed after EVERY chunk in chunked mode
- Flushed every 500 frames in streaming mode
- Console shows GPU usage: `GPU after flush: 0.42GB`

### Browser Memory
- Preview grid reduced to 256x256 (was 512x512)
- Only 16 preview faces stored
- Minimal tensor data returned

### RAM
- Streaming: ~500 MB constant
- Chunked: Up to `memory_threshold_percent` of system RAM

---

## Backend Comparison

### InsightFace (Non-Commercial Only)
```
+ Best accuracy
+ Neural network embeddings (512-dim ArcFace)
+ GPU accelerated
- Non-commercial license only!
```

### MediaPipe (Commercial OK)
```
+ Good accuracy  
+ Apache 2.0 license - commercial safe
+ Google maintained
- CPU only (but still fast)
- Uses histogram-based embeddings (less precise)
```

### OpenCV Cascade (Commercial OK)
```
+ Always available (built into OpenCV)
+ BSD license - commercial safe
+ Very fast
- Lower accuracy
- Uses histogram-based embeddings
```

---

## Similarity Threshold Guide

| Backend | Same Person | Different Person | Recommended |
|---------|-------------|------------------|-------------|
| insightface | 0.65-0.85 | 0.20-0.40 | 0.55-0.65 |
| mediapipe | 0.70-0.90 | 0.40-0.60 | 0.65-0.75 |
| opencv_cascade | 0.75-0.95 | 0.50-0.70 | 0.70-0.80 |

**Use Face Matcher node to find optimal threshold for your specific use case!**

---

## Output Structure

```
ComfyUI/output/
├── face_extract_001/
│   ├── aligned/
│   │   ├── 00000000_0.png
│   │   ├── 00000024_0.png
│   │   └── ...
│   ├── masks/
│   │   ├── 00000000_0_mask.png
│   │   └── ...
│   └── extraction_log.json
```

The `extraction_log.json` includes:
- Backend used and its license type
- Aspect ratio preservation info
- Similarity scores for each face
- Processing statistics

---

## Workflows

| Workflow | Description |
|----------|-------------|
| 01_basic_extraction | Single reference + video path |
| 02_multi_reference | Multiple references for robust matching |
| 03_threshold_tuning | Find optimal similarity threshold |
| 04_deaging_pipeline | Source + destination extraction |

---

## Troubleshooting

### "InsightFace not available"
```bash
pip install insightface onnxruntime-gpu
```

### "MediaPipe not available"  
```bash
pip install mediapipe
```

### GPU memory keeps growing
- Use `streaming` mode instead of `chunked`
- Or reduce `memory_threshold_percent`

### Faces look squashed
- Update to v4 - aspect ratio is now preserved

### Browser tab using too much memory
- v4 returns smaller preview (256x256)
- Consider closing preview nodes if not needed

---

## License

MIT License (this ComfyUI node code)

**IMPORTANT**: The detection backends have their own licenses:
- InsightFace: Non-Commercial only
- MediaPipe: Apache 2.0
- OpenCV: BSD

Your use of this tool must comply with the license of the backend you choose.

---

## Credits

- [InsightFace](https://github.com/deepinsight/insightface) - Face detection/recognition (non-commercial)
- [MediaPipe](https://github.com/google/mediapipe) - Face detection (Apache 2.0)
- [OpenCV](https://opencv.org/) - Computer vision library (BSD)

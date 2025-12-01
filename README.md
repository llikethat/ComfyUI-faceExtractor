# DFL Face Extractor for ComfyUI

Automated face extraction with reference-based matching for DeepFaceLab de-aging and face swap workflows.

## Features

- **Unified Face Extractor**: Single node with dual input modes
- **GPU Accelerated**: Face detection and embedding on CUDA (falls back to CPU)
- **Large Video Support**: Streaming mode processes videos of any length
- **Memory Efficient**: Chunked processing with periodic GPU memory cleanup
- **Reference-based matching**: Provide reference image(s), automatically extract matching faces
- **Auto-incrementing output**: Each run creates a new numbered folder (no overwrites!)
- **DFL-compatible output**: Generates aligned faces and masks ready for DeepFaceLab

## GPU vs CPU Processing

| Component | Device | Notes |
|-----------|--------|-------|
| InsightFace Detection | **GPU** | Uses ONNX Runtime + CUDA |
| InsightFace Embedding | **GPU** | ArcFace model on CUDA |
| Image I/O | CPU | OpenCV read/write |
| Mask Generation | CPU | OpenCV operations |
| Video Decoding | CPU | OpenCV VideoCapture |

The node automatically detects CUDA availability and uses GPU when possible.

## Installation

### 1. Clone to ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/llikethat/ComfyUI_faceExtractor.git
```

### 2. Install dependencies

```bash
cd comfyui_faceExtractor
pip install -r requirements.txt
```

### 3. Install VHS (Video Helper Suite) for video loading

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
```

## Nodes

### DFL Reference Embedding

Computes face embeddings from one or more reference images.

| Input | Type | Description |
|-------|------|-------------|
| reference_images | IMAGE | Single image or batch of images |
| detection_backend | dropdown | "insightface" (recommended) or "opencv_cascade" |
| min_confidence | float | Minimum detection confidence (default: 0.5) |

| Output | Type | Description |
|--------|------|-------------|
| reference_embedding | FACE_EMBEDDING | Averaged embedding for matching |

**Tips:**
- Use 3-10 reference images for more robust matching
- Include different angles, expressions, and lighting

---

### DFL Face Extractor

**Unified extractor with two input modes for handling any video size!**

#### Input Mode 1: IMAGE (for short clips)
- Connect `VHS_LoadVideo`, `LoadImage`, or `LoadImageSequence`
- All frames loaded to RAM first
- Best for clips under ~1000 frames

#### Input Mode 2: video_path (for large videos) ⭐ RECOMMENDED
- Provide direct path to video file as STRING
- **Streaming mode** - processes one frame at a time
- Works with hours of footage
- Minimal RAM usage

| Input | Type | Description |
|-------|------|-------------|
| reference_embedding | FACE_EMBEDDING | From DFL Reference Embedding node |
| images | IMAGE | (Optional) For short clips via VHS_LoadVideo |
| **video_path** | STRING | **(Optional) Direct path to video file - USE FOR LARGE VIDEOS** |
| similarity_threshold | float | Match threshold 0.3-0.95 (default: 0.6) |
| margin_factor | float | Padding around face (default: 0.4) |
| output_size | int | Output image size in pixels (default: 512) |
| max_faces_per_frame | int | Max matching faces per frame (default: 1) |
| **frame_skip** | int | Process every Nth frame (default: 1) |
| **start_frame** | int | Start processing at this frame (default: 0) |
| **end_frame** | int | Stop at this frame, -1 = all (default: -1) |
| output_prefix | string | Folder name prefix (default: "dfl_extract") |
| save_to_disk | bool | Save faces to disk (default: true) |
| save_debug_info | bool | Save extraction log JSON (default: true) |
| detection_backend | dropdown | "insightface" or "opencv_cascade" |
| **clear_gpu_every_n_frames** | int | GPU memory cleanup interval (default: 500) |

| Output | Type | Description |
|--------|------|-------------|
| extracted_faces | IMAGE | Batch of extracted face images |
| masks | MASK | Corresponding face masks |
| output_path | STRING | Path to output folder |
| extracted_count | INT | Number of faces extracted |
| preview_grid | IMAGE | Preview grid of first 16 faces |

**Auto-incrementing output:**
- First run: `ComfyUI/output/dfl_extract_001/`
- Second run: `ComfyUI/output/dfl_extract_002/`
- etc.

---

### DFL Face Matcher

Compare two faces and get similarity score. Useful for threshold tuning.

| Input | Type | Description |
|-------|------|-------------|
| image_a | IMAGE | First face image |
| image_b | IMAGE | Second face image |

| Output | Type | Description |
|--------|------|-------------|
| similarity | FLOAT | Cosine similarity (0-1) |
| match_info | STRING | Human-readable match assessment |

---

### DFL Batch Saver

Save extracted faces to custom location (optional - Face Extractor already saves).

---

## Workflows

### 01 - Basic Extraction
```
[LoadImage: Reference] → [DFL Reference Embedding]
                                    ↓
[VHS_LoadVideo: Source] → [DFL Face Extractor] → [Preview]
                                    ↓
                          output/dfl_extract_001/
```

### 02 - Multi-Reference (Recommended)
```
[Ref 1] ─┐
[Ref 2] ─┼→ [ImageBatch] → [DFL Reference Embedding] → [DFL Face Extractor]
[Ref 3] ─┘      (averaged embedding = more robust)
```

### 03 - Threshold Tuning
```
[Reference] ─┬→ [Face Matcher] → Similarity: 0.78 ✓
[Same Person]┘

[Reference] ─┬→ [Face Matcher] → Similarity: 0.32 ✗
[Different] ─┘

→ Choose threshold between scores (e.g., 0.55)
```

### 04 - Complete De-aging Pipeline
```
🔵 SOURCE (Young):
[Young Ref] → [Embedding] → [Extractor] → output/data_src_001/

🔴 DESTINATION (Old):
[Old Ref] → [Embedding] → [Extractor] → output/data_dst_001/

Then copy to DFL workspace and train!
```

### 05 - Batch Multi-Video
```
[Single Reference] → [Embedding] ─┬→ [Extractor] → movie1_faces_001/
                                  ├→ [Extractor] → movie2_faces_001/
                                  └→ [Extractor] → movie3_faces_001/

Merge: cp output/*_faces_*/aligned/* workspace/data_src/aligned/
```

## Directory Structure

After extraction:
```
ComfyUI/output/
├── dfl_extract_001/
│   ├── aligned/
│   │   ├── 00000000_0.png
│   │   ├── 00000001_0.png
│   │   └── ...
│   ├── masks/
│   │   ├── 00000000_0_mask.png
│   │   └── ...
│   └── extraction_log.json
├── dfl_extract_002/
│   └── ...
```

## Recommended Settings

| Use Case | Threshold | Notes |
|----------|-----------|-------|
| Strict matching | 0.7-0.8 | May miss valid faces |
| **General use** | **0.55-0.65** | **Recommended** |
| Inclusive | 0.45-0.55 | More faces, some false positives |
| Too loose | <0.4 | Will match wrong people |

## Tips

1. **Run threshold tuning first** (Workflow 03)
2. **Use multi-reference** for production extractions
3. **Source dataset**: Need 10,000+ faces for good de-aging
4. **Keep settings consistent** between source and destination (margin_factor, output_size)

## Handling Large Videos

For videos longer than a few minutes, **always use `video_path` input** instead of IMAGE input:

```
❌ VHS_LoadVideo (2-hour movie) → DFL Face Extractor
   Problem: Loads ALL frames to RAM = crash!

✅ DFL Face Extractor with video_path="/path/to/movie.mp4"
   Solution: Streams one frame at a time = works!
```

### Memory Usage Comparison

| Method | 10 min video (24fps) | 2 hour movie |
|--------|---------------------|--------------|
| IMAGE input | ~86 GB RAM | ~1 TB RAM ❌ |
| video_path | ~500 MB RAM | ~500 MB RAM ✅ |

### Recommended Settings for Long Videos

```
video_path: /path/to/movie.mp4
frame_skip: 2-5 (faster extraction, still get thousands of faces)
clear_gpu_every_n_frames: 500 (prevents GPU memory buildup)
start_frame: 0 (or specific scene start)
end_frame: -1 (all frames, or specific scene end)
```

## License

MIT License

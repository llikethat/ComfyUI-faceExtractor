# DFL Face Extractor for ComfyUI

Automated face extraction with reference-based matching for DeepFaceLab de-aging and face swap workflows.

## Features

- **Unified Face Extractor**: Single node works with any IMAGE input (video, image sequence, single image)
- **Reference-based matching**: Provide reference image(s), automatically extract matching faces
- **Auto-incrementing output**: Each run creates a new numbered folder (no overwrites!)
- **DFL-compatible output**: Generates aligned faces and masks ready for DeepFaceLab
- **InsightFace backend**: High-accuracy face detection and recognition using ArcFace embeddings

## Installation

### 1. Clone to ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/comfyui_dfl_extractor.git
```

### 2. Install dependencies

```bash
cd comfyui_dfl_extractor
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

**Unified extractor - works with any IMAGE input!**

| Input | Type | Description |
|-------|------|-------------|
| images | IMAGE | From VHS_LoadVideo, LoadImage, LoadImageSequence, etc. |
| reference_embedding | FACE_EMBEDDING | From DFL Reference Embedding node |
| similarity_threshold | float | Match threshold 0.3-0.95 (default: 0.6) |
| margin_factor | float | Padding around face (default: 0.4) |
| output_size | int | Output image size in pixels (default: 512) |
| max_faces_per_frame | int | Max matching faces per frame (default: 1) |
| output_prefix | string | Folder name prefix (default: "dfl_extract") |
| save_to_disk | bool | Save faces to disk (default: true) |
| save_debug_info | bool | Save extraction log JSON (default: true) |
| detection_backend | dropdown | "insightface" or "opencv_cascade" |

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

## License

MIT License

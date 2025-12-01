"""
DFL Face Extractor Nodes for ComfyUI
Version 3.0 - Memory-aware chunked processing

Features:
- Built-in video loading with memory-aware chunking (no VHS dependency for large videos)
- True frame-by-frame streaming for unlimited video length
- Configurable memory threshold
- GPU acceleration with automatic memory management
"""

import os
import gc
import cv2
import numpy as np
import torch
import psutil
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Generator, Union
from datetime import datetime
import json
import re
import time

# ComfyUI imports
import folder_paths
import comfy.utils

# InsightFace for detection and embedding
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[DFL Extractor] InsightFace not found, will use fallback methods")


# =============================================================================
# Memory Management Utilities
# =============================================================================

def get_system_memory_info() -> dict:
    """Get current system memory usage"""
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / (1024**3),
        'available_gb': mem.available / (1024**3),
        'used_gb': mem.used / (1024**3),
        'percent_used': mem.percent
    }


def get_gpu_memory_info() -> dict:
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - allocated,
            'device_name': torch.cuda.get_device_name(0)
        }
    return {'available': False}


def clear_memory():
    """Aggressively clear both CPU and GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def estimate_frame_memory_mb(width: int, height: int, dtype_bytes: int = 3) -> float:
    """Estimate memory usage per frame in MB"""
    # Raw frame + processing overhead (roughly 2x for safety)
    return (width * height * dtype_bytes * 2) / (1024**2)


def calculate_safe_chunk_size(
    video_width: int,
    video_height: int,
    memory_threshold_percent: float = 75.0,
    min_chunk: int = 10,
    max_chunk: int = 500
) -> int:
    """Calculate safe chunk size based on available memory"""
    mem_info = get_system_memory_info()
    available_mb = mem_info['available_gb'] * 1024
    
    # Calculate how much memory we can use
    usable_mb = available_mb * (memory_threshold_percent / 100.0)
    
    # Estimate per-frame memory
    frame_mb = estimate_frame_memory_mb(video_width, video_height)
    
    # Calculate chunk size with safety margin
    safe_chunk = int(usable_mb / frame_mb / 2)  # Divide by 2 for safety
    
    # Clamp to reasonable bounds
    return max(min_chunk, min(safe_chunk, max_chunk))


# =============================================================================
# Face Detection Backend
# =============================================================================

class FaceDetectorBackend:
    """
    Unified face detection backend with GPU acceleration.
    Singleton pattern to avoid reinitializing the model.
    """
    
    _instance = None
    _current_backend = None
    
    def __new__(cls, backend: str = "insightface", device: str = "cuda"):
        if cls._instance is None or cls._current_backend != backend:
            cls._instance = super().__new__(cls)
            cls._current_backend = backend
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, backend: str = "insightface", device: str = "cuda"):
        if self._initialized:
            return
            
        self.backend = backend
        self.device = device
        self.detector = None
        self._initialize()
        self._initialized = True
    
    def _initialize(self):
        if self.backend == "insightface" and INSIGHTFACE_AVAILABLE:
            providers = ['CPUExecutionProvider']
            ctx_id = -1
            
            if torch.cuda.is_available() and self.device == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                ctx_id = 0
                gpu_info = get_gpu_memory_info()
                print(f"[DFL Extractor] Using GPU: {gpu_info['device_name']} ({gpu_info['total_gb']:.1f} GB)")
            else:
                print("[DFL Extractor] Using CPU for face detection")
            
            self.detector = FaceAnalysis(
                name='buffalo_l',
                providers=providers
            )
            self.detector.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self.using_gpu = ctx_id >= 0
        else:
            self.backend = "opencv_cascade"
            self.using_gpu = False
            print("[DFL Extractor] Using OpenCV cascade (CPU only)")
            
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """Detect faces in image"""
        if self.backend == "insightface" and self.detector:
            faces = self.detector.get(image)
            results = []
            for face in faces:
                results.append({
                    'bbox': face.bbox.astype(int).tolist(),
                    'landmarks': face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else face.kps,
                    'embedding': face.embedding,
                    'confidence': float(face.det_score),
                })
            return results
        else:
            return self._opencv_detect(image)
    
    def _opencv_detect(self, image: np.ndarray) -> List[dict]:
        """Fallback OpenCV cascade detector"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        results = []
        for (x, y, w, h) in faces:
            results.append({
                'bbox': [x, y, x+w, y+h],
                'landmarks': None,
                'embedding': None,
                'confidence': 0.9
            })
        return results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def extract_face_with_margin(
    image: np.ndarray,
    bbox: List[int],
    margin_factor: float = 0.4,
    target_size: Tuple[int, int] = (512, 512)
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Extract face region with margin and create mask"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    face_w = x2 - x1
    face_h = y2 - y1
    
    margin_w = int(face_w * margin_factor)
    margin_h = int(face_h * margin_factor)
    
    x1_exp = max(0, x1 - margin_w)
    y1_exp = max(0, y1 - margin_h)
    x2_exp = min(w, x2 + margin_w)
    y2_exp = min(h, y2 + margin_h)
    
    face_region = image[y1_exp:y2_exp, x1_exp:x2_exp].copy()
    
    mask = np.zeros((y2_exp - y1_exp, x2_exp - x1_exp), dtype=np.uint8)
    
    rel_x1 = x1 - x1_exp
    rel_y1 = y1 - y1_exp
    rel_x2 = x2 - x1_exp
    rel_y2 = y2 - y1_exp
    
    center_x = (rel_x1 + rel_x2) // 2
    center_y = (rel_y1 + rel_y2) // 2
    axes = ((rel_x2 - rel_x1) // 2, (rel_y2 - rel_y1) // 2)
    cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 11)
    
    if target_size:
        face_region = cv2.resize(face_region, target_size, interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    transform_info = {
        'original_bbox': bbox,
        'expanded_bbox': [x1_exp, y1_exp, x2_exp, y2_exp],
        'source_size': (w, h),
        'margin_factor': margin_factor
    }
    
    return face_region, mask, transform_info


def get_next_output_folder(base_path: Path, prefix: str = "dfl_extract") -> Path:
    """Get next available output folder with auto-incrementing number"""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    existing = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    
    for item in base_path.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                existing.append(int(match.group(1)))
    
    next_num = max(existing, default=0) + 1
    new_folder = base_path / f"{prefix}_{next_num:03d}"
    new_folder.mkdir(parents=True, exist_ok=True)
    
    return new_folder


def create_preview_grid(images: List[np.ndarray], grid_size: int = 4, cell_size: int = 128) -> np.ndarray:
    """Create a grid preview of extracted faces"""
    if not images:
        return np.zeros((512, 512, 3), dtype=np.uint8)
    
    grid = np.zeros((grid_size * cell_size, grid_size * cell_size, 3), dtype=np.uint8)
    
    for idx, img in enumerate(images[:grid_size * grid_size]):
        row = idx // grid_size
        col = idx % grid_size
        resized = cv2.resize(img, (cell_size, cell_size))
        y_start = row * cell_size
        x_start = col * cell_size
        grid[y_start:y_start + cell_size, x_start:x_start + cell_size] = resized
    
    return grid


# =============================================================================
# Video Processing Classes
# =============================================================================

class VideoInfo:
    """Container for video metadata"""
    def __init__(self, path: str):
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_seconds = self.total_frames / self.fps if self.fps > 0 else 0
        cap.release()
    
    def __str__(self):
        return f"Video: {self.total_frames} frames, {self.width}x{self.height}, {self.fps:.2f} fps, {self.duration_seconds:.1f}s"


class ChunkedVideoProcessor:
    """
    Memory-aware chunked video processor.
    Loads and processes video in chunks based on available memory.
    """
    
    def __init__(
        self,
        video_path: str,
        memory_threshold_percent: float = 75.0,
        frame_skip: int = 1,
        start_frame: int = 0,
        end_frame: int = -1
    ):
        self.video_path = video_path
        self.memory_threshold = memory_threshold_percent
        self.frame_skip = frame_skip
        self.start_frame = start_frame
        
        self.video_info = VideoInfo(video_path)
        self.end_frame = end_frame if end_frame > 0 else self.video_info.total_frames
        
        # Calculate optimal chunk size
        self.chunk_size = calculate_safe_chunk_size(
            self.video_info.width,
            self.video_info.height,
            memory_threshold_percent
        )
        
        # Calculate total frames to process
        self.frames_to_process = []
        for i in range(self.start_frame, self.end_frame, self.frame_skip):
            self.frames_to_process.append(i)
        
        self.total_frames_to_process = len(self.frames_to_process)
        
        print(f"[DFL Extractor] {self.video_info}")
        print(f"[DFL Extractor] Processing {self.total_frames_to_process} frames (skip={frame_skip})")
        print(f"[DFL Extractor] Chunk size: {self.chunk_size} frames")
        print(f"[DFL Extractor] Memory threshold: {memory_threshold_percent}%")
    
    def process_chunks(
        self,
        detector: FaceDetectorBackend,
        ref_embedding: np.ndarray,
        similarity_threshold: float,
        margin_factor: float,
        output_size: int,
        max_faces_per_frame: int,
        aligned_dir: Path,
        masks_dir: Path,
        extraction_log: List[dict],
        preview_faces: List[np.ndarray],
        pbar
    ) -> int:
        """
        Process video in memory-aware chunks.
        Returns total number of faces extracted.
        """
        total_extracted = 0
        chunk_idx = 0
        frame_list_idx = 0
        
        while frame_list_idx < len(self.frames_to_process):
            # Determine chunk boundaries
            chunk_start_idx = frame_list_idx
            chunk_end_idx = min(frame_list_idx + self.chunk_size, len(self.frames_to_process))
            chunk_frames = self.frames_to_process[chunk_start_idx:chunk_end_idx]
            
            mem_before = get_system_memory_info()
            print(f"[DFL Extractor] Processing chunk {chunk_idx + 1}: frames {chunk_frames[0]}-{chunk_frames[-1]} "
                  f"(Memory: {mem_before['percent_used']:.1f}% used)")
            
            # Load chunk of frames
            frames_data = self._load_frame_chunk(chunk_frames)
            
            # Process each frame in chunk
            for i, (frame_idx, frame) in enumerate(frames_data):
                # Detect faces
                faces = detector.detect_faces(frame)
                
                # Score by similarity
                scored_faces = []
                for face in faces:
                    emb = face.get('embedding')
                    if emb is not None:
                        sim = cosine_similarity(ref_embedding, emb)
                        if sim >= similarity_threshold:
                            scored_faces.append((sim, face))
                
                # Sort and select top matches
                scored_faces.sort(key=lambda x: x[0], reverse=True)
                selected_faces = scored_faces[:max_faces_per_frame]
                
                for face_idx, (sim_score, face) in enumerate(selected_faces):
                    face_img, mask, _ = extract_face_with_margin(
                        frame,
                        face['bbox'],
                        margin_factor=margin_factor,
                        target_size=(output_size, output_size)
                    )
                    
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    
                    # Keep preview faces (first 16)
                    if len(preview_faces) < 16:
                        preview_faces.append(face_rgb.copy())
                    
                    # Save to disk immediately
                    face_filename = f"{frame_idx:08d}_{face_idx}.png"
                    mask_filename = f"{frame_idx:08d}_{face_idx}_mask.png"
                    cv2.imwrite(str(aligned_dir / face_filename), face_img)
                    cv2.imwrite(str(masks_dir / mask_filename), mask)
                    
                    extraction_log.append({
                        'frame_idx': frame_idx,
                        'face_idx': face_idx,
                        'timestamp': frame_idx / self.video_info.fps if self.video_info.fps > 0 else 0,
                        'similarity': float(sim_score),
                        'bbox': face['bbox'],
                        'confidence': face['confidence'],
                        'filename': face_filename
                    })
                    
                    total_extracted += 1
                
                pbar.update(1)
                
                # Clear frame reference
                frame = None
            
            # Clear chunk data and force garbage collection
            frames_data = None
            clear_memory()
            
            mem_after = get_system_memory_info()
            print(f"[DFL Extractor] Chunk {chunk_idx + 1} complete. "
                  f"Extracted: {total_extracted} faces. "
                  f"Memory: {mem_after['percent_used']:.1f}% used")
            
            frame_list_idx = chunk_end_idx
            chunk_idx += 1
        
        return total_extracted
    
    def _load_frame_chunk(self, frame_indices: List[int]) -> List[Tuple[int, np.ndarray]]:
        """Load specific frames from video"""
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append((frame_idx, frame))
        
        cap.release()
        return frames


class StreamingVideoProcessor:
    """
    True streaming processor - processes one frame at a time.
    Minimal memory usage regardless of video length.
    """
    
    def __init__(
        self,
        video_path: str,
        frame_skip: int = 1,
        start_frame: int = 0,
        end_frame: int = -1
    ):
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.start_frame = start_frame
        
        self.video_info = VideoInfo(video_path)
        self.end_frame = end_frame if end_frame > 0 else self.video_info.total_frames
        
        # Calculate total frames to process
        self.total_frames_to_process = len(range(start_frame, self.end_frame, frame_skip))
        
        print(f"[DFL Extractor] {self.video_info}")
        print(f"[DFL Extractor] STREAMING MODE - Processing {self.total_frames_to_process} frames")
        print(f"[DFL Extractor] Memory usage: ~constant (1 frame at a time)")
    
    def frame_generator(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generator that yields one frame at a time"""
        cap = cv2.VideoCapture(self.video_path)
        
        if self.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        frame_idx = self.start_frame
        
        while frame_idx < self.end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            if (frame_idx - self.start_frame) % self.frame_skip == 0:
                yield frame_idx, frame
            
            frame_idx += 1
        
        cap.release()


# =============================================================================
# ComfyUI Nodes
# =============================================================================

class DFLReferenceEmbedding:
    """
    Load reference face image(s) and compute embeddings.
    Uses GPU for face detection and embedding computation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE",),
            },
            "optional": {
                "detection_backend": (["insightface", "opencv_cascade"], {"default": "insightface"}),
                "min_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("FACE_EMBEDDING",)
    RETURN_NAMES = ("reference_embedding",)
    FUNCTION = "compute_embedding"
    CATEGORY = "DFL Extractor"
    
    def compute_embedding(
        self,
        reference_images: torch.Tensor,
        detection_backend: str = "insightface",
        min_confidence: float = 0.5
    ):
        detector = FaceDetectorBackend(backend=detection_backend)
        embeddings = []
        
        if len(reference_images.shape) == 4:
            images = reference_images
        else:
            images = reference_images.unsqueeze(0)
        
        print(f"[DFL Extractor] Computing embeddings from {images.shape[0]} reference images...")
        
        for i in range(images.shape[0]):
            img = images[i].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            faces = detector.detect_faces(img)
            
            for face in faces:
                if face['confidence'] >= min_confidence:
                    emb = face.get('embedding')
                    if emb is not None:
                        embeddings.append(emb)
        
        if not embeddings:
            raise ValueError("No faces detected in reference images with sufficient confidence")
        
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        
        print(f"[DFL Extractor] Created embedding from {len(embeddings)} face(s)")
        
        return ({
            'embedding': avg_embedding,
            'num_references': len(embeddings),
            'detector_backend': detection_backend,
            'using_gpu': detector.using_gpu,
        },)


class DFLFaceExtractor:
    """
    Memory-aware face extractor with built-in video loading.
    
    NO NEED FOR VHS_LoadVideo - provide video path directly!
    
    Processing modes:
    1. STREAMING (default): One frame at a time - works with ANY video length
    2. CHUNKED: Load frames in memory-aware batches - faster but uses more RAM
    
    For IMAGE input (from other nodes): Processes in memory-aware batches
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_embedding": ("FACE_EMBEDDING",),
                "video_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                # Alternative IMAGE input (for compatibility, but video_path preferred)
                "images": ("IMAGE",),
                # Processing settings
                "similarity_threshold": ("FLOAT", {"default": 0.6, "min": 0.3, "max": 0.95, "step": 0.05}),
                "margin_factor": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.05}),
                "output_size": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64}),
                "max_faces_per_frame": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                # Video settings
                "frame_skip": ("INT", {"default": 1, "min": 1, "max": 60, "step": 1}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999999, "step": 1}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": 9999999, "step": 1}),
                # Memory management
                "processing_mode": (["streaming", "chunked"], {"default": "streaming"}),
                "memory_threshold_percent": ("FLOAT", {"default": 75.0, "min": 30.0, "max": 90.0, "step": 5.0}),
                # Output settings
                "output_prefix": ("STRING", {"default": "dfl_extract", "multiline": False}),
                "save_debug_info": ("BOOLEAN", {"default": True}),
                # Performance
                "detection_backend": (["insightface", "opencv_cascade"], {"default": "insightface"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "IMAGE")
    RETURN_NAMES = ("output_path", "extracted_count", "preview_grid")
    FUNCTION = "extract_faces"
    CATEGORY = "DFL Extractor"
    OUTPUT_NODE = True
    
    def extract_faces(
        self,
        reference_embedding: dict,
        video_path: str = "",
        images: torch.Tensor = None,
        similarity_threshold: float = 0.6,
        margin_factor: float = 0.4,
        output_size: int = 512,
        max_faces_per_frame: int = 1,
        frame_skip: int = 1,
        start_frame: int = 0,
        end_frame: int = -1,
        processing_mode: str = "streaming",
        memory_threshold_percent: float = 75.0,
        output_prefix: str = "dfl_extract",
        save_debug_info: bool = True,
        detection_backend: str = "insightface"
    ):
        # Determine input mode
        use_video_path = video_path and os.path.exists(video_path)
        use_image_input = images is not None and images.numel() > 0
        
        if not use_video_path and not use_image_input:
            raise ValueError("Provide either 'video_path' to a video file or connect 'images' input")
        
        if use_video_path and use_image_input:
            print("[DFL Extractor] Both inputs provided - using video_path (recommended for large videos)")
            use_image_input = False
        
        # Show memory status
        mem_info = get_system_memory_info()
        print(f"[DFL Extractor] System Memory: {mem_info['available_gb']:.1f} GB available / {mem_info['total_gb']:.1f} GB total")
        
        # Initialize detector
        detector = FaceDetectorBackend(backend=detection_backend)
        ref_emb = reference_embedding['embedding']
        
        # Setup output directory
        output_base = Path(folder_paths.get_output_directory())
        output_path = get_next_output_folder(output_base, output_prefix)
        
        aligned_dir = output_path / "aligned"
        masks_dir = output_path / "masks"
        aligned_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        extraction_log = []
        preview_faces = []
        total_extracted = 0
        
        start_time = time.time()
        
        # ================================================================
        # VIDEO PATH PROCESSING (Recommended for large videos)
        # ================================================================
        if use_video_path:
            if processing_mode == "streaming":
                # TRUE STREAMING - One frame at a time
                processor = StreamingVideoProcessor(
                    video_path, frame_skip, start_frame, end_frame
                )
                
                pbar = comfy.utils.ProgressBar(processor.total_frames_to_process)
                
                for frame_idx, frame in processor.frame_generator():
                    # Detect faces
                    faces = detector.detect_faces(frame)
                    
                    # Score by similarity
                    scored_faces = []
                    for face in faces:
                        emb = face.get('embedding')
                        if emb is not None:
                            sim = cosine_similarity(ref_emb, emb)
                            if sim >= similarity_threshold:
                                scored_faces.append((sim, face))
                    
                    # Select top matches
                    scored_faces.sort(key=lambda x: x[0], reverse=True)
                    selected_faces = scored_faces[:max_faces_per_frame]
                    
                    for face_idx, (sim_score, face) in enumerate(selected_faces):
                        face_img, mask, _ = extract_face_with_margin(
                            frame,
                            face['bbox'],
                            margin_factor=margin_factor,
                            target_size=(output_size, output_size)
                        )
                        
                        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        
                        if len(preview_faces) < 16:
                            preview_faces.append(face_rgb.copy())
                        
                        # Save immediately
                        face_filename = f"{frame_idx:08d}_{face_idx}.png"
                        mask_filename = f"{frame_idx:08d}_{face_idx}_mask.png"
                        cv2.imwrite(str(aligned_dir / face_filename), face_img)
                        cv2.imwrite(str(masks_dir / mask_filename), mask)
                        
                        extraction_log.append({
                            'frame_idx': frame_idx,
                            'face_idx': face_idx,
                            'timestamp': frame_idx / processor.video_info.fps if processor.video_info.fps > 0 else 0,
                            'similarity': float(sim_score),
                            'bbox': face['bbox'],
                            'confidence': face['confidence'],
                            'filename': face_filename
                        })
                        
                        total_extracted += 1
                    
                    pbar.update(1)
                    
                    # Periodic cleanup
                    if len(extraction_log) % 500 == 0 and len(extraction_log) > 0:
                        clear_memory()
                
            else:
                # CHUNKED PROCESSING - Memory-aware batches
                processor = ChunkedVideoProcessor(
                    video_path, memory_threshold_percent, frame_skip, start_frame, end_frame
                )
                
                pbar = comfy.utils.ProgressBar(processor.total_frames_to_process)
                
                total_extracted = processor.process_chunks(
                    detector, ref_emb, similarity_threshold, margin_factor,
                    output_size, max_faces_per_frame, aligned_dir, masks_dir,
                    extraction_log, preview_faces, pbar
                )
        
        # ================================================================
        # IMAGE INPUT PROCESSING (From VHS or other nodes)
        # ================================================================
        else:
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            
            num_frames = images.shape[0]
            print(f"[DFL Extractor] Processing {num_frames} images from IMAGE input")
            print(f"[DFL Extractor] WARNING: For large videos, use video_path instead of VHS_LoadVideo!")
            
            # Calculate batch size based on memory
            frame_memory_mb = estimate_frame_memory_mb(images.shape[2], images.shape[1])
            batch_size = calculate_safe_chunk_size(
                images.shape[2], images.shape[1], 
                memory_threshold_percent, min_chunk=1, max_chunk=100
            )
            
            print(f"[DFL Extractor] Processing in batches of {batch_size} frames")
            
            pbar = comfy.utils.ProgressBar(num_frames)
            
            for batch_start in range(0, num_frames, batch_size):
                batch_end = min(batch_start + batch_size, num_frames)
                
                for frame_idx in range(batch_start, batch_end):
                    img = images[frame_idx].cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    faces = detector.detect_faces(img_bgr)
                    
                    scored_faces = []
                    for face in faces:
                        emb = face.get('embedding')
                        if emb is not None:
                            sim = cosine_similarity(ref_emb, emb)
                            if sim >= similarity_threshold:
                                scored_faces.append((sim, face))
                    
                    scored_faces.sort(key=lambda x: x[0], reverse=True)
                    selected_faces = scored_faces[:max_faces_per_frame]
                    
                    for face_idx, (sim_score, face) in enumerate(selected_faces):
                        face_img, mask, _ = extract_face_with_margin(
                            img_bgr,
                            face['bbox'],
                            margin_factor=margin_factor,
                            target_size=(output_size, output_size)
                        )
                        
                        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        
                        if len(preview_faces) < 16:
                            preview_faces.append(face_rgb)
                        
                        face_filename = f"{frame_idx:08d}_{face_idx}.png"
                        mask_filename = f"{frame_idx:08d}_{face_idx}_mask.png"
                        cv2.imwrite(str(aligned_dir / face_filename), face_img)
                        cv2.imwrite(str(masks_dir / mask_filename), mask)
                        
                        extraction_log.append({
                            'frame_idx': frame_idx,
                            'face_idx': face_idx,
                            'similarity': float(sim_score),
                            'bbox': face['bbox'],
                            'confidence': face['confidence'],
                            'filename': face_filename
                        })
                        
                        total_extracted += 1
                    
                    pbar.update(1)
                
                # Clear memory after each batch
                clear_memory()
        
        # ================================================================
        # Finalize
        # ================================================================
        elapsed_time = time.time() - start_time
        
        # Save extraction log
        if save_debug_info:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': elapsed_time,
                'processing_mode': processing_mode if use_video_path else "image_batch",
                'source': video_path if use_video_path else "image_input",
                'extracted_count': total_extracted,
                'settings': {
                    'similarity_threshold': similarity_threshold,
                    'margin_factor': margin_factor,
                    'output_size': output_size,
                    'max_faces_per_frame': max_faces_per_frame,
                    'frame_skip': frame_skip,
                    'memory_threshold_percent': memory_threshold_percent,
                    'detection_backend': detection_backend
                },
                'reference_info': {
                    'num_references': reference_embedding.get('num_references', 1)
                },
                'system_info': {
                    'gpu': get_gpu_memory_info(),
                    'memory': get_system_memory_info()
                },
                'extractions': extraction_log
            }
            with open(output_path / "extraction_log.json", 'w') as f:
                json.dump(log_data, f, indent=2)
        
        # Create preview grid
        if preview_faces:
            grid = create_preview_grid(preview_faces)
            grid_tensor = torch.from_numpy(grid).float() / 255.0
            grid_tensor = grid_tensor.unsqueeze(0)
        else:
            grid_tensor = torch.zeros(1, 512, 512, 3)
        
        # Final cleanup
        clear_memory()
        
        print(f"[DFL Extractor] ✓ Extracted {total_extracted} faces to {output_path}")
        print(f"[DFL Extractor] ✓ Time: {elapsed_time:.1f}s ({total_extracted/elapsed_time:.1f} faces/sec)")
        
        return (str(output_path), total_extracted, grid_tensor)


class DFLFaceMatcher:
    """Compare two faces and get similarity score"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "optional": {
                "detection_backend": (["insightface", "opencv_cascade"], {"default": "insightface"}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("similarity", "match_info")
    FUNCTION = "compare_faces"
    CATEGORY = "DFL Extractor"
    
    def compare_faces(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        detection_backend: str = "insightface"
    ):
        detector = FaceDetectorBackend(backend=detection_backend)
        
        img_a = image_a[0].cpu().numpy() if len(image_a.shape) == 4 else image_a.cpu().numpy()
        img_b = image_b[0].cpu().numpy() if len(image_b.shape) == 4 else image_b.cpu().numpy()
        
        img_a = (img_a * 255).astype(np.uint8)
        img_b = (img_b * 255).astype(np.uint8)
        
        img_a_bgr = cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR)
        img_b_bgr = cv2.cvtColor(img_b, cv2.COLOR_RGB2BGR)
        
        faces_a = detector.detect_faces(img_a_bgr)
        faces_b = detector.detect_faces(img_b_bgr)
        
        if not faces_a or not faces_b:
            return (0.0, "No faces detected in one or both images")
        
        emb_a = faces_a[0].get('embedding')
        emb_b = faces_b[0].get('embedding')
        
        if emb_a is None or emb_b is None:
            return (0.0, "Failed to compute embeddings")
        
        similarity = cosine_similarity(emb_a, emb_b)
        
        match_level = "No match"
        if similarity >= 0.7:
            match_level = "Strong match (same person)"
        elif similarity >= 0.5:
            match_level = "Possible match"
        elif similarity >= 0.3:
            match_level = "Weak similarity"
        
        gpu_info = get_gpu_memory_info()
        gpu_str = f"\nUsing GPU: {gpu_info.get('device_name', 'N/A')}" if gpu_info.get('device_name') else "\nUsing CPU"
        
        info = f"Similarity: {similarity:.4f}\nMatch level: {match_level}{gpu_str}"
        
        return (float(similarity), info)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "DFLReferenceEmbedding": DFLReferenceEmbedding,
    "DFLFaceExtractor": DFLFaceExtractor,
    "DFLFaceMatcher": DFLFaceMatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DFLReferenceEmbedding": "DFL Reference Embedding",
    "DFLFaceExtractor": "DFL Face Extractor",
    "DFLFaceMatcher": "DFL Face Matcher",
}

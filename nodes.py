"""
DFL Face Extractor Nodes for ComfyUI
Optimized version with unified extractor node
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import json
import hashlib
import re

# ComfyUI imports
import folder_paths
import comfy.utils

# We'll use insightface for detection and embedding
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[DFL Extractor] InsightFace not found, will use fallback methods")


class FaceEmbeddingCache:
    """Cache for face embeddings to avoid recomputation"""
    _cache = {}
    
    @classmethod
    def get_key(cls, image_data: np.ndarray) -> str:
        return hashlib.md5(image_data.tobytes()).hexdigest()
    
    @classmethod
    def get(cls, key: str) -> Optional[np.ndarray]:
        return cls._cache.get(key)
    
    @classmethod
    def set(cls, key: str, embedding: np.ndarray):
        cls._cache[key] = embedding


class FaceDetectorBackend:
    """Unified face detection backend supporting multiple libraries"""
    
    _instance = None
    _current_backend = None
    
    def __new__(cls, backend: str = "insightface", device: str = "cuda"):
        # Singleton pattern to avoid reinitializing detector
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
            self.detector = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.detector.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))
        else:
            self.backend = "opencv_cascade"
            
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in image.
        Returns list of dicts with: bbox, landmarks, embedding, confidence
        """
        if self.backend == "insightface" and self.detector:
            faces = self.detector.get(image)
            results = []
            for face in faces:
                results.append({
                    'bbox': face.bbox.astype(int).tolist(),
                    'landmarks': face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else face.kps,
                    'embedding': face.embedding,
                    'confidence': float(face.det_score),
                    'age': getattr(face, 'age', None),
                    'gender': getattr(face, 'gender', None)
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
    
    def get_embedding(self, image: np.ndarray, face_info: dict) -> Optional[np.ndarray]:
        """Get face embedding for a detected face"""
        if face_info.get('embedding') is not None:
            return face_info['embedding']
        
        if self.backend == "insightface" and self.detector:
            bbox = face_info['bbox']
            x1, y1, x2, y2 = [int(c) for c in bbox]
            face_crop = image[y1:y2, x1:x2]
            if face_crop.size == 0:
                return None
            faces = self.detector.get(face_crop)
            if faces:
                return faces[0].embedding
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def extract_face_with_margin(
    image: np.ndarray,
    bbox: List[int],
    margin_factor: float = 0.4,
    target_size: Tuple[int, int] = (512, 512)
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Extract face region with margin and create mask.
    Returns: (face_image, mask, transform_info)
    """
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
    """
    Get next available output folder with auto-incrementing number.
    Creates folders like: dfl_extract_001, dfl_extract_002, etc.
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Find existing folders with this prefix
    existing = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    
    for item in base_path.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                existing.append(int(match.group(1)))
    
    # Get next number
    next_num = max(existing, default=0) + 1
    
    # Create new folder
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


class DFLReferenceEmbedding:
    """
    Load reference face image(s) and compute embeddings.
    Supports single image or batch of images for more robust matching.
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
        
        # Handle batch of images
        if len(reference_images.shape) == 4:
            images = reference_images
        else:
            images = reference_images.unsqueeze(0)
        
        for i in range(images.shape[0]):
            img = images[i].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            faces = detector.detect_faces(img)
            
            for face in faces:
                if face['confidence'] >= min_confidence:
                    emb = face.get('embedding')
                    if emb is None:
                        emb = detector.get_embedding(img, face)
                    if emb is not None:
                        embeddings.append(emb)
        
        if not embeddings:
            raise ValueError("No faces detected in reference images with sufficient confidence")
        
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        
        return ({
            'embedding': avg_embedding,
            'num_references': len(embeddings),
            'detector_backend': detection_backend
        },)


class DFLFaceExtractor:
    """
    Unified face extractor - works with any IMAGE input.
    Accepts images from LoadImage, VHS_LoadVideo, LoadImageSequence, etc.
    Auto-creates output folders with incrementing numbers.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference_embedding": ("FACE_EMBEDDING",),
            },
            "optional": {
                "similarity_threshold": ("FLOAT", {"default": 0.6, "min": 0.3, "max": 0.95, "step": 0.05}),
                "margin_factor": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.05}),
                "output_size": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64}),
                "max_faces_per_frame": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "output_prefix": ("STRING", {"default": "dfl_extract", "multiline": False}),
                "save_to_disk": ("BOOLEAN", {"default": True}),
                "save_debug_info": ("BOOLEAN", {"default": True}),
                "detection_backend": (["insightface", "opencv_cascade"], {"default": "insightface"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT", "IMAGE")
    RETURN_NAMES = ("extracted_faces", "masks", "output_path", "extracted_count", "preview_grid")
    FUNCTION = "extract_faces"
    CATEGORY = "DFL Extractor"
    OUTPUT_NODE = True
    
    def extract_faces(
        self,
        images: torch.Tensor,
        reference_embedding: dict,
        similarity_threshold: float = 0.6,
        margin_factor: float = 0.4,
        output_size: int = 512,
        max_faces_per_frame: int = 1,
        output_prefix: str = "dfl_extract",
        save_to_disk: bool = True,
        save_debug_info: bool = True,
        detection_backend: str = "insightface"
    ):
        # Initialize detector
        detector = FaceDetectorBackend(backend=detection_backend)
        ref_emb = reference_embedding['embedding']
        
        # Handle batch dimensions
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        
        num_frames = images.shape[0]
        
        # Setup output directory
        output_base = Path(folder_paths.get_output_directory())
        output_path = get_next_output_folder(output_base, output_prefix)
        
        if save_to_disk:
            aligned_dir = output_path / "aligned"
            masks_dir = output_path / "masks"
            aligned_dir.mkdir(exist_ok=True)
            masks_dir.mkdir(exist_ok=True)
        
        extracted_faces = []
        extracted_masks = []
        extraction_log = []
        preview_faces = []
        
        pbar = comfy.utils.ProgressBar(num_frames)
        
        for frame_idx in range(num_frames):
            # Convert tensor to numpy
            img = images[frame_idx].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Detect faces
            faces = detector.detect_faces(img_bgr)
            
            # Score faces by similarity
            scored_faces = []
            for face in faces:
                emb = face.get('embedding')
                if emb is None:
                    emb = detector.get_embedding(img_bgr, face)
                if emb is not None:
                    sim = cosine_similarity(ref_emb, emb)
                    if sim >= similarity_threshold:
                        scored_faces.append((sim, face))
            
            # Sort by similarity and take top matches
            scored_faces.sort(key=lambda x: x[0], reverse=True)
            selected_faces = scored_faces[:max_faces_per_frame]
            
            for face_idx, (sim_score, face) in enumerate(selected_faces):
                # Extract face with margin
                face_img, mask, transform_info = extract_face_with_margin(
                    img_bgr,
                    face['bbox'],
                    margin_factor=margin_factor,
                    target_size=(output_size, output_size)
                )
                
                # Convert to RGB
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                extracted_faces.append(face_rgb)
                extracted_masks.append(mask)
                
                # Save preview (first 16)
                if len(preview_faces) < 16:
                    preview_faces.append(face_rgb)
                
                # Save to disk
                if save_to_disk:
                    face_filename = f"{frame_idx:08d}_{face_idx}.png"
                    mask_filename = f"{frame_idx:08d}_{face_idx}_mask.png"
                    
                    cv2.imwrite(str(aligned_dir / face_filename), face_img)
                    cv2.imwrite(str(masks_dir / mask_filename), mask)
                
                # Log extraction
                extraction_log.append({
                    'frame_idx': frame_idx,
                    'face_idx': face_idx,
                    'similarity': float(sim_score),
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'filename': f"{frame_idx:08d}_{face_idx}.png" if save_to_disk else None
                })
            
            pbar.update(1)
        
        # Save extraction log
        if save_to_disk and save_debug_info:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'total_frames': num_frames,
                'extracted_count': len(extracted_faces),
                'settings': {
                    'similarity_threshold': similarity_threshold,
                    'margin_factor': margin_factor,
                    'output_size': output_size,
                    'max_faces_per_frame': max_faces_per_frame,
                    'detection_backend': detection_backend
                },
                'reference_info': {
                    'num_references': reference_embedding.get('num_references', 1)
                },
                'extractions': extraction_log
            }
            with open(output_path / "extraction_log.json", 'w') as f:
                json.dump(log_data, f, indent=2)
        
        # Create output tensors
        if extracted_faces:
            faces_np = np.stack(extracted_faces, axis=0)
            masks_np = np.stack(extracted_masks, axis=0)
            
            faces_tensor = torch.from_numpy(faces_np).float() / 255.0
            masks_tensor = torch.from_numpy(masks_np).float() / 255.0
        else:
            faces_tensor = torch.zeros(1, output_size, output_size, 3)
            masks_tensor = torch.zeros(1, output_size, output_size)
        
        # Create preview grid
        if preview_faces:
            grid = create_preview_grid(preview_faces)
            grid_tensor = torch.from_numpy(grid).float() / 255.0
            grid_tensor = grid_tensor.unsqueeze(0)
        else:
            grid_tensor = torch.zeros(1, 512, 512, 3)
        
        return (faces_tensor, masks_tensor, str(output_path), len(extracted_faces), grid_tensor)


class DFLFaceMatcher:
    """
    Simple face matching node - outputs similarity score between two faces.
    Useful for debugging and threshold tuning.
    """
    
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
        
        if emb_a is None:
            emb_a = detector.get_embedding(img_a_bgr, faces_a[0])
        if emb_b is None:
            emb_b = detector.get_embedding(img_b_bgr, faces_b[0])
        
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
        
        info = f"Similarity: {similarity:.4f}\nMatch level: {match_level}"
        
        return (float(similarity), info)


class DFLBatchSaver:
    """
    Save extracted faces in DFL-compatible directory structure.
    Use this if you want to save to a custom location or merge datasets.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "faces": ("IMAGE",),
                "masks": ("MASK",),
            },
            "optional": {
                "output_prefix": ("STRING", {"default": "dfl_dataset", "multiline": False}),
                "dataset_type": (["source", "destination"], {"default": "source"}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "filename_prefix": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("output_path", "saved_count")
    FUNCTION = "save_dataset"
    CATEGORY = "DFL Extractor"
    OUTPUT_NODE = True
    
    def save_dataset(
        self,
        faces: torch.Tensor,
        masks: torch.Tensor,
        output_prefix: str = "dfl_dataset",
        dataset_type: str = "source",
        start_index: int = 0,
        filename_prefix: str = ""
    ):
        output_base = Path(folder_paths.get_output_directory())
        output_path = get_next_output_folder(output_base, output_prefix)
        
        # DFL directory structure
        aligned_dir = output_path / "aligned"
        aligned_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to numpy
        faces_np = (faces.cpu().numpy() * 255).astype(np.uint8)
        masks_np = (masks.cpu().numpy() * 255).astype(np.uint8)
        
        saved_count = 0
        
        for i in range(faces_np.shape[0]):
            idx = start_index + i
            prefix = f"{filename_prefix}_" if filename_prefix else ""
            
            face_filename = f"{prefix}{idx:08d}.png"
            mask_filename = f"{prefix}{idx:08d}_mask.png"
            
            face_bgr = cv2.cvtColor(faces_np[i], cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(str(aligned_dir / face_filename), face_bgr)
            
            if len(masks_np.shape) == 3:
                mask = masks_np[i]
            else:
                mask = masks_np
            cv2.imwrite(str(aligned_dir / mask_filename), mask)
            
            saved_count += 1
        
        # Save info file
        info = {
            'dataset_type': dataset_type,
            'count': saved_count,
            'start_index': start_index,
            'timestamp': datetime.now().isoformat()
        }
        with open(output_path / "dataset_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        return (str(output_path), saved_count)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "DFLReferenceEmbedding": DFLReferenceEmbedding,
    "DFLFaceExtractor": DFLFaceExtractor,
    "DFLFaceMatcher": DFLFaceMatcher,
    "DFLBatchSaver": DFLBatchSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DFLReferenceEmbedding": "DFL Reference Embedding",
    "DFLFaceExtractor": "DFL Face Extractor",
    "DFLFaceMatcher": "DFL Face Matcher",
    "DFLBatchSaver": "DFL Dataset Saver",
}

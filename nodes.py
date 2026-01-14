"""
Face Extractor Nodes for ComfyUI
Version 4.0 - Generic Face Extraction Tool

Features:
- Memory-aware chunked processing with GPU flush
- Aspect ratio preservation (no squashing)
- Minimal browser memory footprint
- Multiple detection backends:
  - InsightFace (NON-COMMERCIAL USE ONLY)
  - MediaPipe (Apache 2.0 - Commercial OK)
  - OpenCV Cascade (BSD - Commercial OK)
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

# Detection backends
INSIGHTFACE_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False
FACENET_AVAILABLE = False
YOLOV8_AVAILABLE = False

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    pass

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    pass

try:
    from ultralytics import YOLO
    YOLOV8_AVAILABLE = True
except ImportError:
    pass


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


def clear_memory(clear_gpu: bool = True):
    """Aggressively clear both CPU and GPU memory"""
    gc.collect()
    if clear_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def estimate_frame_memory_mb(width: int, height: int, dtype_bytes: int = 3) -> float:
    """Estimate memory usage per frame in MB"""
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
    usable_mb = available_mb * (memory_threshold_percent / 100.0)
    frame_mb = estimate_frame_memory_mb(video_width, video_height)
    safe_chunk = int(usable_mb / frame_mb / 2)
    return max(min_chunk, min(safe_chunk, max_chunk))


# =============================================================================
# Face Detection Backends
# =============================================================================

class InsightFaceBackend:
    """
    InsightFace backend - HIGH QUALITY but NON-COMMERCIAL USE ONLY!
    License: https://github.com/deepinsight/insightface#license
    """
    
    LICENSE = "NON-COMMERCIAL"
    
    def __init__(self, device: str = "cuda"):
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("InsightFace not installed. Run: pip install insightface onnxruntime-gpu")
        
        providers = ['CPUExecutionProvider']
        ctx_id = -1
        
        if torch.cuda.is_available() and device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = 0
        
        self.detector = FaceAnalysis(name='buffalo_l', providers=providers)
        self.detector.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.using_gpu = ctx_id >= 0
        self.name = "insightface"
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        faces = self.detector.get(image)
        results = []
        for face in faces:
            results.append({
                'bbox': face.bbox.astype(int).tolist(),
                'landmarks': face.kps if hasattr(face, 'kps') else None,
                'embedding': face.embedding,
                'confidence': float(face.det_score),
            })
        return results


class MediaPipeBackend:
    """
    MediaPipe backend - Commercial use OK (Apache 2.0 License)
    Note: Does not provide face embeddings, only detection
    """
    
    LICENSE = "COMMERCIAL_OK"
    
    def __init__(self, device: str = "cuda"):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe not installed. Run: pip install mediapipe")
        
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5
        )
        self.using_gpu = False  # MediaPipe uses CPU by default
        self.name = "mediapipe"
        
        # For embeddings, we'll use a simple approach based on face region
        # This is less accurate than InsightFace but allows commercial use
        self._embedding_size = 128
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        # MediaPipe expects RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        results = self.detector.process(rgb_image)
        faces = []
        
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Generate simple embedding from face region
                embedding = self._compute_simple_embedding(image, [x1, y1, x2, y2])
                
                faces.append({
                    'bbox': [x1, y1, x2, y2],
                    'landmarks': None,
                    'embedding': embedding,
                    'confidence': float(detection.score[0]),
                })
        
        return faces
    
    def _compute_simple_embedding(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Compute a simple embedding based on face region histogram and features.
        Less accurate than neural embeddings but allows commercial use.
        """
        x1, y1, x2, y2 = bbox
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return np.zeros(self._embedding_size)
        
        # Resize to standard size
        face_resized = cv2.resize(face_region, (64, 64))
        
        # Convert to LAB color space for better color representation
        lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
        
        # Compute histogram features
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([lab], [i], None, [32], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-8)
            hist_features.extend(hist)
        
        # Add spatial features (downsampled grayscale)
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        spatial = cv2.resize(gray, (8, 8)).flatten() / 255.0
        
        # Combine features
        embedding = np.concatenate([hist_features, spatial])
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Pad or truncate to embedding size
        if len(embedding) < self._embedding_size:
            embedding = np.pad(embedding, (0, self._embedding_size - len(embedding)))
        else:
            embedding = embedding[:self._embedding_size]
        
        return embedding.astype(np.float32)
    
    def __del__(self):
        if hasattr(self, 'detector'):
            self.detector.close()


class OpenCVCascadeBackend:
    """
    OpenCV Cascade backend - Commercial use OK (BSD License)
    Basic detection, no embeddings (uses histogram-based matching)
    """
    
    LICENSE = "COMMERCIAL_OK"
    
    def __init__(self, device: str = "cuda"):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.using_gpu = False
        self.name = "opencv_cascade"
        self._embedding_size = 128
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        results = []
        for (x, y, w, h) in faces:
            bbox = [x, y, x + w, y + h]
            embedding = self._compute_simple_embedding(image, bbox)
            results.append({
                'bbox': bbox,
                'landmarks': None,
                'embedding': embedding,
                'confidence': 0.9
            })
        return results
    
    def _compute_simple_embedding(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Same histogram-based embedding as MediaPipe backend"""
        x1, y1, x2, y2 = bbox
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return np.zeros(self._embedding_size)
        
        face_resized = cv2.resize(face_region, (64, 64))
        lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
        
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([lab], [i], None, [32], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-8)
            hist_features.extend(hist)
        
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        spatial = cv2.resize(gray, (8, 8)).flatten() / 255.0
        
        embedding = np.concatenate([hist_features, spatial])
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        if len(embedding) < self._embedding_size:
            embedding = np.pad(embedding, (0, self._embedding_size - len(embedding)))
        else:
            embedding = embedding[:self._embedding_size]
        
        return embedding.astype(np.float32)


class FaceNetBackend:
    """
    FaceNet backend using facenet-pytorch - COMMERCIAL USE OK (MIT License)
    Uses MTCNN for detection and InceptionResnetV1 for embeddings.
    High quality neural network embeddings similar to InsightFace.
    
    License: MIT - https://github.com/timesler/facenet-pytorch/blob/master/LICENSE
    """
    
    LICENSE = "COMMERCIAL_OK"
    
    def __init__(self, device: str = "cuda"):
        if not FACENET_AVAILABLE:
            raise RuntimeError("FaceNet not installed. Run: pip install facenet-pytorch")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == "cuda" else 'cpu')
        
        # MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=True  # Detect all faces
        )
        
        # InceptionResnetV1 for embeddings (pretrained on VGGFace2)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.using_gpu = self.device.type == 'cuda'
        self.name = "facenet"
        
        if self.using_gpu:
            print(f"[Face Extractor] FaceNet using GPU: {torch.cuda.get_device_name(0)}")
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Convert to PIL for MTCNN
        pil_image = Image.fromarray(rgb_image)
        
        # Detect faces - returns boxes and probabilities
        boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
        
        results = []
        
        if boxes is not None:
            # Get face crops for embedding computation
            faces_cropped = self.mtcnn(pil_image)
            
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob is None:
                    continue
                    
                x1, y1, x2, y2 = [int(c) for c in box]
                
                # Clamp to image bounds
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Compute embedding if face was cropped successfully
                embedding = None
                if faces_cropped is not None:
                    if len(faces_cropped.shape) == 3:
                        # Single face
                        face_tensor = faces_cropped.unsqueeze(0).to(self.device)
                    else:
                        # Multiple faces
                        if i < faces_cropped.shape[0]:
                            face_tensor = faces_cropped[i].unsqueeze(0).to(self.device)
                        else:
                            face_tensor = None
                    
                    if face_tensor is not None:
                        with torch.no_grad():
                            embedding = self.resnet(face_tensor).cpu().numpy().flatten()
                
                # Get landmarks for this face if available
                face_landmarks = None
                if landmarks is not None and i < len(landmarks):
                    face_landmarks = landmarks[i]
                
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'landmarks': face_landmarks,
                    'embedding': embedding,
                    'confidence': float(prob),
                })
        
        return results


class YOLOv8Backend:
    """
    YOLOv8 backend using Ultralytics - AGPL-3.0 License
    
    ⚠️  COMMERCIAL USE REQUIRES ULTRALYTICS LICENSE!
    For commercial closed-source applications, you must purchase a license from Ultralytics.
    See: https://ultralytics.com/license
    
    Uses YOLOv8 for face detection with FaceNet-style embeddings.
    """
    
    LICENSE = "AGPL-3.0 (Commercial license required for closed-source)"
    
    def __init__(self, device: str = "cuda"):
        if not YOLOV8_AVAILABLE:
            raise RuntimeError("YOLOv8 not installed. Run: pip install ultralytics")
        
        self.device = 'cuda' if torch.cuda.is_available() and device == "cuda" else 'cpu'
        
        # Load YOLOv8 face model
        # First try yolov8n-face, fall back to general yolov8n
        try:
            # Try to load face-specific model if available
            self.model = YOLO('yolov8n-face.pt')
            self.is_face_model = True
        except:
            # Fall back to general model - will detect 'person' class
            self.model = YOLO('yolov8n.pt')
            self.is_face_model = False
            print("[Face Extractor] YOLOv8 face model not found, using general model")
        
        self.model.to(self.device)
        self.using_gpu = self.device == 'cuda'
        self.name = "yolov8"
        self._embedding_size = 512
        
        # Try to use FaceNet for embeddings if available (better quality)
        self._facenet_embedder = None
        if FACENET_AVAILABLE:
            try:
                facenet_device = torch.device(self.device)
                self._facenet_embedder = InceptionResnetV1(pretrained='vggface2').eval().to(facenet_device)
                print("[Face Extractor] YOLOv8 using FaceNet for embeddings")
            except:
                pass
        
        if self.using_gpu:
            print(f"[Face Extractor] YOLOv8 using GPU: {torch.cuda.get_device_name(0)}")
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        # Convert BGR to RGB for YOLO
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Run YOLO detection
        results = self.model(rgb_image, verbose=False)
        
        faces = []
        h, w = image.shape[:2]
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for i, box in enumerate(boxes):
                # Get class - for face model it's face, for general model look for person (class 0)
                cls = int(box.cls[0])
                
                # Filter: face model detects faces directly, general model we use class 0 (person)
                # For person detection, we'll try to find face region within the person bbox
                if not self.is_face_model and cls != 0:
                    continue
                
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # For person detection, estimate face region (upper portion)
                if not self.is_face_model:
                    person_h = y2 - y1
                    # Face is typically in top 30% of person bbox
                    y2 = y1 + int(person_h * 0.35)
                    # Narrow the width slightly
                    person_w = x2 - x1
                    x1 = x1 + int(person_w * 0.15)
                    x2 = x2 - int(person_w * 0.15)
                
                # Compute embedding
                embedding = self._compute_embedding(image, [x1, y1, x2, y2])
                
                faces.append({
                    'bbox': [x1, y1, x2, y2],
                    'landmarks': None,
                    'embedding': embedding,
                    'confidence': conf,
                })
        
        return faces
    
    def _compute_embedding(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Compute face embedding using FaceNet if available, otherwise histogram-based"""
        x1, y1, x2, y2 = bbox
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return np.zeros(self._embedding_size)
        
        # Use FaceNet embedder if available
        if self._facenet_embedder is not None:
            try:
                # Resize and normalize for FaceNet
                face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (160, 160))
                
                # Convert to tensor
                face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float()
                face_tensor = (face_tensor - 127.5) / 128.0  # Normalize
                face_tensor = face_tensor.unsqueeze(0).to(next(self._facenet_embedder.parameters()).device)
                
                with torch.no_grad():
                    embedding = self._facenet_embedder(face_tensor).cpu().numpy().flatten()
                
                return embedding
            except Exception as e:
                pass  # Fall back to histogram
        
        # Histogram-based fallback
        face_resized = cv2.resize(face_region, (64, 64))
        lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
        
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([lab], [i], None, [32], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-8)
            hist_features.extend(hist)
        
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        spatial = cv2.resize(gray, (8, 8)).flatten() / 255.0
        
        embedding = np.concatenate([hist_features, spatial])
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Pad to embedding size
        if len(embedding) < self._embedding_size:
            embedding = np.pad(embedding, (0, self._embedding_size - len(embedding)))
        else:
            embedding = embedding[:self._embedding_size]
        
        return embedding.astype(np.float32)


class FaceDetectorBackend:
    """
    Unified face detection backend factory.
    Singleton pattern to avoid reinitializing models.
    """
    
    _instances = {}
    
    @classmethod
    def get_backend(cls, backend: str = "facenet", device: str = "cuda"):
        key = f"{backend}_{device}"
        
        if key not in cls._instances:
            # FaceNet - MIT License (Commercial OK) - RECOMMENDED
            if backend == "facenet":
                if not FACENET_AVAILABLE:
                    print("[Face Extractor] FaceNet not available, falling back to MediaPipe")
                    backend = "mediapipe"
                else:
                    print("[Face Extractor] ✓ FaceNet: Commercial use OK (MIT License)")
                    cls._instances[key] = FaceNetBackend(device)
                    return cls._instances[key]
            
            # YOLOv8 - AGPL-3.0 (Commercial requires license)
            if backend == "yolov8":
                if not YOLOV8_AVAILABLE:
                    print("[Face Extractor] YOLOv8 not available, falling back to FaceNet")
                    backend = "facenet"
                    if not FACENET_AVAILABLE:
                        backend = "mediapipe"
                else:
                    print("[Face Extractor] ⚠️  YOLOv8: AGPL-3.0 - Commercial use requires Ultralytics license!")
                    cls._instances[key] = YOLOv8Backend(device)
                    return cls._instances[key]
            
            # InsightFace - Non-commercial only
            if backend == "insightface":
                if not INSIGHTFACE_AVAILABLE:
                    print("[Face Extractor] InsightFace not available, falling back to FaceNet")
                    backend = "facenet"
                    if not FACENET_AVAILABLE:
                        backend = "mediapipe"
                else:
                    print("[Face Extractor] ⚠️  InsightFace: NON-COMMERCIAL USE ONLY!")
                    cls._instances[key] = InsightFaceBackend(device)
                    return cls._instances[key]
            
            # MediaPipe - Apache 2.0 (Commercial OK)
            if backend == "mediapipe":
                if not MEDIAPIPE_AVAILABLE:
                    print("[Face Extractor] MediaPipe not available, falling back to OpenCV")
                    backend = "opencv_cascade"
                else:
                    print("[Face Extractor] ✓ MediaPipe: Commercial use OK (Apache 2.0)")
                    cls._instances[key] = MediaPipeBackend(device)
                    return cls._instances[key]
            
            # OpenCV Cascade - BSD (Commercial OK)
            if backend == "opencv_cascade":
                print("[Face Extractor] ✓ OpenCV Cascade: Commercial use OK (BSD)")
                cls._instances[key] = OpenCVCascadeBackend(device)
                return cls._instances[key]
            
            raise ValueError(f"Unknown backend: {backend}")
        
        return cls._instances[key]
    
    @classmethod
    def get_available_backends(cls) -> List[str]:
        """Return list of available backends, ordered by recommendation"""
        backends = ["opencv_cascade"]  # Always available
        if MEDIAPIPE_AVAILABLE:
            backends.insert(0, "mediapipe")
        if INSIGHTFACE_AVAILABLE:
            backends.insert(0, "insightface")
        if YOLOV8_AVAILABLE:
            backends.insert(0, "yolov8")
        if FACENET_AVAILABLE:
            backends.insert(0, "facenet")  # FaceNet first if available (MIT + high quality)
        return backends


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def extract_face_preserve_aspect(
    image: np.ndarray,
    bbox: List[int],
    margin_factor: float = 0.4,
    target_size: int = 512,
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Extract face region with margin, PRESERVING ASPECT RATIO.
    Pads with black to make square if needed.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    face_w = x2 - x1
    face_h = y2 - y1
    
    # Add margin
    margin_w = int(face_w * margin_factor)
    margin_h = int(face_h * margin_factor)
    
    x1_exp = max(0, x1 - margin_w)
    y1_exp = max(0, y1 - margin_h)
    x2_exp = min(w, x2 + margin_w)
    y2_exp = min(h, y2 + margin_h)
    
    # Extract region
    face_region = image[y1_exp:y2_exp, x1_exp:x2_exp].copy()
    region_h, region_w = face_region.shape[:2]
    
    # Calculate scale to fit in target_size while preserving aspect ratio
    scale = target_size / max(region_w, region_h)
    new_w = int(region_w * scale)
    new_h = int(region_h * scale)
    
    # Resize preserving aspect ratio
    face_resized = cv2.resize(face_region, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create square canvas and center the face
    canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    
    # Calculate centering offsets
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    
    # Place resized face on canvas
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = face_resized
    
    # Create mask (elliptical, centered on the actual face)
    mask = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # Calculate ellipse parameters relative to the canvas
    face_center_x = x_offset + new_w // 2
    face_center_y = y_offset + new_h // 2
    
    # Scale the original face bbox to get ellipse axes
    orig_face_w_scaled = int((face_w / region_w) * new_w)
    orig_face_h_scaled = int((face_h / region_h) * new_h)
    
    axes = (orig_face_w_scaled // 2 + int(margin_factor * orig_face_w_scaled // 2),
            orig_face_h_scaled // 2 + int(margin_factor * orig_face_h_scaled // 2))
    
    cv2.ellipse(mask, (face_center_x, face_center_y), axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 11)
    
    transform_info = {
        'original_bbox': bbox,
        'expanded_bbox': [x1_exp, y1_exp, x2_exp, y2_exp],
        'source_size': (w, h),
        'region_size': (region_w, region_h),
        'scaled_size': (new_w, new_h),
        'offset': (x_offset, y_offset),
        'scale': scale,
        'margin_factor': margin_factor,
        'aspect_ratio_preserved': True
    }
    
    return canvas, mask, transform_info


def get_next_output_folder(base_path: Path, prefix: str = "face_extract") -> Path:
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


def create_preview_grid(
    images: List[np.ndarray], 
    grid_size: int = 4, 
    cell_size: int = 64  # Reduced from 128 to minimize browser memory
) -> np.ndarray:
    """Create a small grid preview of extracted faces"""
    if not images:
        return np.zeros((256, 256, 3), dtype=np.uint8)  # Smaller default
    
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
# Video Processing
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
    """Memory-aware chunked video processor with GPU flush after each chunk"""
    
    def __init__(
        self,
        video_path: str,
        memory_threshold_percent: float = 75.0,
        frame_skip: int = 1,
        start_frame: int = 0,
        end_frame: int = -1,
        manual_chunk_size: int = 0  # 0 = auto-calculate
    ):
        self.video_path = video_path
        self.memory_threshold = memory_threshold_percent
        self.frame_skip = frame_skip
        self.start_frame = start_frame
        
        self.video_info = VideoInfo(video_path)
        self.end_frame = end_frame if end_frame > 0 else self.video_info.total_frames
        
        # Use manual chunk size if provided, otherwise auto-calculate
        if manual_chunk_size > 0:
            self.chunk_size = manual_chunk_size
            print(f"[Face Extractor] Using manual chunk size: {self.chunk_size}")
        else:
            self.chunk_size = calculate_safe_chunk_size(
                self.video_info.width,
                self.video_info.height,
                memory_threshold_percent
            )
            print(f"[Face Extractor] Auto chunk size: {self.chunk_size} (based on {memory_threshold_percent}% memory threshold)")
        
        self.frames_to_process = list(range(self.start_frame, self.end_frame, self.frame_skip))
        self.total_frames_to_process = len(self.frames_to_process)
        
        print(f"[Face Extractor] {self.video_info}")
        print(f"[Face Extractor] Processing {self.total_frames_to_process} frames (skip={frame_skip})")
    
    def process_chunks(
        self,
        detector,
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
        """Process video in memory-aware chunks with GPU flush after each"""
        total_extracted = 0
        chunk_idx = 0
        frame_list_idx = 0
        
        while frame_list_idx < len(self.frames_to_process):
            chunk_start_idx = frame_list_idx
            chunk_end_idx = min(frame_list_idx + self.chunk_size, len(self.frames_to_process))
            chunk_frames = self.frames_to_process[chunk_start_idx:chunk_end_idx]
            
            mem_before = get_system_memory_info()
            gpu_before = get_gpu_memory_info()
            print(f"[Face Extractor] Chunk {chunk_idx + 1}: frames {chunk_frames[0]}-{chunk_frames[-1]} "
                  f"(RAM: {mem_before['percent_used']:.1f}%, GPU: {gpu_before.get('allocated_gb', 0):.2f}GB)")
            
            # Load chunk
            frames_data = self._load_frame_chunk(chunk_frames)
            
            # Process each frame
            for frame_idx, frame in frames_data:
                faces = detector.detect_faces(frame)
                
                scored_faces = []
                for face in faces:
                    emb = face.get('embedding')
                    if emb is not None:
                        sim = cosine_similarity(ref_embedding, emb)
                        if sim >= similarity_threshold:
                            scored_faces.append((sim, face))
                
                scored_faces.sort(key=lambda x: x[0], reverse=True)
                selected_faces = scored_faces[:max_faces_per_frame]
                
                for face_idx, (sim_score, face) in enumerate(selected_faces):
                    face_img, mask, transform_info = extract_face_preserve_aspect(
                        frame,
                        face['bbox'],
                        margin_factor=margin_factor,
                        target_size=output_size
                    )
                    
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    
                    if len(preview_faces) < 16:
                        preview_faces.append(face_rgb.copy())
                    
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
                        'aspect_preserved': transform_info.get('aspect_ratio_preserved', True),
                        'filename': face_filename
                    })
                    
                    total_extracted += 1
                
                pbar.update(1)
            
            # IMPORTANT: Clear memory after EVERY chunk
            frames_data = None
            clear_memory(clear_gpu=True)  # Explicit GPU flush
            
            gpu_after = get_gpu_memory_info()
            print(f"[Face Extractor] Chunk {chunk_idx + 1} done. "
                  f"Faces: {total_extracted}, GPU after flush: {gpu_after.get('allocated_gb', 0):.2f}GB")
            
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
    """True streaming - one frame at a time, minimal memory"""
    
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
        self.total_frames_to_process = len(range(start_frame, self.end_frame, frame_skip))
        
        print(f"[Face Extractor] {self.video_info}")
        print(f"[Face Extractor] STREAMING MODE - {self.total_frames_to_process} frames")
    
    def frame_generator(self) -> Generator[Tuple[int, np.ndarray], None, None]:
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

class FaceReferenceEmbedding:
    """
    Compute face embeddings from reference images.
    
    BACKEND LICENSE INFO:
    - insightface: NON-COMMERCIAL USE ONLY
    - mediapipe: Commercial OK (Apache 2.0)
    - opencv_cascade: Commercial OK (BSD)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        backends = FaceDetectorBackend.get_available_backends()
        return {
            "required": {
                "reference_images": ("IMAGE",),
            },
            "optional": {
                "detection_backend": (backends, {"default": backends[0]}),
                "min_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("FACE_EMBEDDING",)
    RETURN_NAMES = ("reference_embedding",)
    FUNCTION = "compute_embedding"
    CATEGORY = "Face Extractor"
    
    def compute_embedding(
        self,
        reference_images: torch.Tensor,
        detection_backend: str = "insightface",
        min_confidence: float = 0.5
    ):
        detector = FaceDetectorBackend.get_backend(backend=detection_backend)
        embeddings = []
        
        if len(reference_images.shape) == 4:
            images = reference_images
        else:
            images = reference_images.unsqueeze(0)
        
        print(f"[Face Extractor] Computing embeddings from {images.shape[0]} reference(s)...")
        
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
            raise ValueError("No faces detected in reference images")
        
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        
        print(f"[Face Extractor] Created embedding from {len(embeddings)} face(s)")
        
        return ({
            'embedding': avg_embedding,
            'num_references': len(embeddings),
            'backend': detection_backend,
            'license': detector.LICENSE,
            'using_gpu': detector.using_gpu,
        },)


class FaceExtractor:
    """
    Memory-efficient face extractor with built-in video loading.
    
    BACKEND LICENSE INFO:
    - insightface: NON-COMMERCIAL USE ONLY  
    - mediapipe: Commercial OK (Apache 2.0)
    - opencv_cascade: Commercial OK (BSD)
    
    For commercial projects, use mediapipe or opencv_cascade!
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        backends = FaceDetectorBackend.get_available_backends()
        return {
            "required": {
                "reference_embedding": ("FACE_EMBEDDING",),
                "video_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "images": ("IMAGE",),
                "similarity_threshold": ("FLOAT", {"default": 0.6, "min": 0.3, "max": 0.95, "step": 0.05}),
                "margin_factor": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.05}),
                "output_size": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64}),
                "max_faces_per_frame": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "frame_skip": ("INT", {"default": 1, "min": 1, "max": 60, "step": 1}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999999, "step": 1}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": 9999999, "step": 1}),
                "processing_mode": (["streaming", "chunked"], {"default": "streaming"}),
                "memory_threshold_percent": ("FLOAT", {"default": 75.0, "min": 30.0, "max": 90.0, "step": 5.0}),
                "chunk_size": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 50}),  # 0 = auto
                "output_prefix": ("STRING", {"default": "face_extract", "multiline": False}),
                "save_debug_info": ("BOOLEAN", {"default": True}),
                "detection_backend": (backends, {"default": backends[0]}),
            }
        }
    
    # Return types include detailed output info
    RETURN_TYPES = ("STRING", "STRING", "INT", "IMAGE")
    RETURN_NAMES = ("output_path", "output_info", "extracted_count", "preview_grid")
    FUNCTION = "extract_faces"
    CATEGORY = "Face Extractor"
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
        chunk_size: int = 0,  # 0 = auto-calculate based on memory
        output_prefix: str = "face_extract",
        save_debug_info: bool = True,
        detection_backend: str = "insightface"
    ):
        use_video_path = video_path and os.path.exists(video_path)
        use_image_input = images is not None and images.numel() > 0
        
        if not use_video_path and not use_image_input:
            raise ValueError("Provide 'video_path' or connect 'images' input")
        
        if use_video_path and use_image_input:
            print("[Face Extractor] Both inputs provided - using video_path")
            use_image_input = False
        
        # Memory status
        mem_info = get_system_memory_info()
        gpu_info = get_gpu_memory_info()
        print(f"[Face Extractor] RAM: {mem_info['available_gb']:.1f}GB free / {mem_info['total_gb']:.1f}GB")
        if gpu_info.get('device_name'):
            print(f"[Face Extractor] GPU: {gpu_info['device_name']} ({gpu_info['free_gb']:.1f}GB free)")
        
        # Initialize detector
        detector = FaceDetectorBackend.get_backend(backend=detection_backend)
        ref_emb = reference_embedding['embedding']
        
        # Setup output
        output_base = Path(folder_paths.get_output_directory())
        output_path = get_next_output_folder(output_base, output_prefix)
        
        aligned_dir = output_path / "aligned"
        masks_dir = output_path / "masks"
        aligned_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        extraction_log = []
        preview_faces = []  # Only store first 16 for preview
        total_extracted = 0
        
        start_time = time.time()
        
        # ================================================================
        # VIDEO PATH PROCESSING
        # ================================================================
        if use_video_path:
            if processing_mode == "streaming":
                processor = StreamingVideoProcessor(video_path, frame_skip, start_frame, end_frame)
                pbar = comfy.utils.ProgressBar(processor.total_frames_to_process)
                
                frames_since_flush = 0
                
                for frame_idx, frame in processor.frame_generator():
                    faces = detector.detect_faces(frame)
                    
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
                        face_img, mask, transform_info = extract_face_preserve_aspect(
                            frame, face['bbox'],
                            margin_factor=margin_factor,
                            target_size=output_size
                        )
                        
                        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        
                        if len(preview_faces) < 16:
                            preview_faces.append(face_rgb.copy())
                        
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
                    frames_since_flush += 1
                    
                    # Periodic GPU flush in streaming mode too
                    if frames_since_flush >= 500:
                        clear_memory(clear_gpu=True)
                        frames_since_flush = 0
                
            else:  # chunked mode
                processor = ChunkedVideoProcessor(
                    video_path, memory_threshold_percent, frame_skip, start_frame, end_frame,
                    manual_chunk_size=chunk_size
                )
                pbar = comfy.utils.ProgressBar(processor.total_frames_to_process)
                
                total_extracted = processor.process_chunks(
                    detector, ref_emb, similarity_threshold, margin_factor,
                    output_size, max_faces_per_frame, aligned_dir, masks_dir,
                    extraction_log, preview_faces, pbar
                )
        
        # ================================================================
        # IMAGE INPUT PROCESSING
        # ================================================================
        else:
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            
            num_frames = images.shape[0]
            print(f"[Face Extractor] Processing {num_frames} images")
            print(f"[Face Extractor] ⚠️  For large videos, use video_path instead!")
            
            # Use manual chunk size if provided, otherwise auto-calculate
            if chunk_size > 0:
                batch_size = chunk_size
                print(f"[Face Extractor] Using manual chunk size: {batch_size}")
            else:
                batch_size = calculate_safe_chunk_size(
                    images.shape[2], images.shape[1],
                    memory_threshold_percent, min_chunk=1, max_chunk=100
                )
                print(f"[Face Extractor] Auto chunk size: {batch_size}")
            
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
                        face_img, mask, transform_info = extract_face_preserve_aspect(
                            img_bgr, face['bbox'],
                            margin_factor=margin_factor,
                            target_size=output_size
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
                
                # GPU flush after each batch
                clear_memory(clear_gpu=True)
        
        # ================================================================
        # Finalize
        # ================================================================
        elapsed_time = time.time() - start_time
        
        if save_debug_info:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': elapsed_time,
                'processing_mode': processing_mode if use_video_path else "image_batch",
                'source': video_path if use_video_path else "image_input",
                'extracted_count': total_extracted,
                'backend': detection_backend,
                'backend_license': detector.LICENSE,
                'settings': {
                    'similarity_threshold': similarity_threshold,
                    'margin_factor': margin_factor,
                    'output_size': output_size,
                    'max_faces_per_frame': max_faces_per_frame,
                    'frame_skip': frame_skip,
                    'processing_mode': processing_mode if use_video_path else "image_batch",
                    'memory_threshold_percent': memory_threshold_percent,
                    'chunk_size': chunk_size if chunk_size > 0 else "auto",
                    'aspect_ratio_preserved': True
                },
                'extractions': extraction_log
            }
            with open(output_path / "extraction_log.json", 'w') as f:
                json.dump(log_data, f, indent=2)
        
        # Create SMALL preview grid (reduced browser memory)
        if preview_faces:
            grid = create_preview_grid(preview_faces, grid_size=4, cell_size=64)
            grid_tensor = torch.from_numpy(grid).float() / 255.0
            grid_tensor = grid_tensor.unsqueeze(0)
        else:
            grid_tensor = torch.zeros(1, 256, 256, 3)
        
        # Final cleanup
        clear_memory(clear_gpu=True)
        
        # Build detailed output info string
        output_info_lines = [
            f"═══════════════════════════════════════════",
            f"EXTRACTION COMPLETE",
            f"═══════════════════════════════════════════",
            f"",
            f"📁 Output Directory:",
            f"   {output_path}",
            f"",
            f"📂 Aligned Faces:",
            f"   {aligned_dir}",
            f"   ({total_extracted} files)",
            f"",
            f"📂 Masks:",
            f"   {masks_dir}",
            f"   ({total_extracted} files)",
            f"",
            f"⏱️  Processing Time: {elapsed_time:.1f}s",
            f"📊 Speed: {total_extracted/max(elapsed_time,1):.1f} faces/sec",
            f"🔧 Backend: {detection_backend} ({detector.LICENSE})",
            f"",
        ]
        
        # Add sample filenames if available
        if extraction_log:
            output_info_lines.append("📄 Sample Files:")
            for entry in extraction_log[:5]:  # Show first 5
                output_info_lines.append(f"   • {entry['filename']} (sim: {entry['similarity']:.3f})")
            if len(extraction_log) > 5:
                output_info_lines.append(f"   ... and {len(extraction_log) - 5} more")
        
        output_info_lines.append(f"═══════════════════════════════════════════")
        
        output_info = "\n".join(output_info_lines)
        
        print(f"[Face Extractor] ✓ Extracted {total_extracted} faces to {output_path}")
        print(f"[Face Extractor] ✓ Time: {elapsed_time:.1f}s ({total_extracted/max(elapsed_time,1):.1f} faces/sec)")
        
        return (str(output_path), output_info, total_extracted, grid_tensor)


class FaceMatcher:
    """Compare two faces for similarity threshold tuning"""
    
    @classmethod
    def INPUT_TYPES(cls):
        backends = FaceDetectorBackend.get_available_backends()
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "optional": {
                "detection_backend": (backends, {"default": backends[0]}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("similarity", "match_info")
    FUNCTION = "compare_faces"
    CATEGORY = "Face Extractor"
    
    def compare_faces(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        detection_backend: str = "insightface"
    ):
        detector = FaceDetectorBackend.get_backend(backend=detection_backend)
        
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
        
        license_info = f"\nBackend: {detection_backend} ({detector.LICENSE})"
        info = f"Similarity: {similarity:.4f}\nMatch level: {match_level}{license_info}"
        
        return (float(similarity), info)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "FaceReferenceEmbedding": FaceReferenceEmbedding,
    "FaceExtractor": FaceExtractor,
    "FaceMatcher": FaceMatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceReferenceEmbedding": "Face Reference Embedding",
    "FaceExtractor": "Face Extractor",
    "FaceMatcher": "Face Matcher",
}

"""
Video Search Implementation

This is the main entry point for your solution. You must implement the
VideoSearch class that inherits from VideoSearchInterface.

See the API documentation in docs/API_CONTRACT.md for details.
"""

"""
Video Search Implementation

This is the main entry point for your solution. You must implement the
VideoSearch class that inherits from VideoSearchInterface.

See the API documentation in docs/API_CONTRACT.md for details.
"""

from typing import List
import sys
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import open_clip
from PIL import Image

# Add parent directory to path for evaluation imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.interface import VideoSearchInterface, SearchResult


class VideoSearch(VideoSearchInterface):
    """
    CLIP-based video search over sampled frames.

    Approach:
    - Sample frames at a fixed interval
    - Encode sampled frames with CLIP
    - Encode text query with CLIP
    - Rank sampled frames by cosine similarity
    - Apply light temporal smoothing to improve stability
    """

    def __init__(self):
        """Initialize the video search system."""
        self.video_path = None
        self.frames = []
        self.timestamps = []
        self.embeddings = None
        self.sampled_frame_numbers = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "ViT-B-32"
        self.pretrained_name = "openai"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained_name,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.stats = {
            "fps": 0.0,
            "memory_mb": 0.0,
            "model_info": {
                "name": f"CLIP {self.model_name}",
                "pretrained": self.pretrained_name,
                "device": self.device,
            },
            "index_time_seconds": 0.0,
            "total_frames": 0,
            "sampled_frames": 0,
            "video_duration_seconds": 0.0,
            "sampling_interval_seconds": 1.0,
        }

    def load_video(self, video_path: str) -> None:
        """
        Load and preprocess a video.

        Args:
            video_path: Path to the video file.

        Raises:
            FileNotFoundError: If video doesn't exist.
            ValueError: If video format is not supported or video cannot be read.
        """
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        supported_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        if video_file.suffix.lower() not in supported_suffixes:
            raise ValueError(f"Unsupported video format: {video_file.suffix}")

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        self.video_path = str(video_file)
        self.frames = []
        self.timestamps = []
        self.sampled_frame_numbers = []
        self.embeddings = None

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = raw_fps if raw_fps and raw_fps > 0 else 25.0

        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = (
            total_frame_count / fps if total_frame_count > 0 and fps > 0 else 0.0
        )

        # Sample 1 frame per second for a good speed/accuracy balance
        frame_interval = max(1, int(round(fps)))

        start_time = time.time()
        embeddings = []
        frame_idx = 0

        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    timestamp = frame_idx / fps

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

                    image_embedding = self.model.encode_image(image_tensor)
                    image_embedding = image_embedding / image_embedding.norm(
                        dim=-1, keepdim=True
                    )

                    embeddings.append(image_embedding.cpu().numpy()[0])
                    self.frames.append(frame.copy())
                    self.timestamps.append(timestamp)
                    self.sampled_frame_numbers.append(frame_idx)

                frame_idx += 1

        cap.release()

        if not embeddings:
            raise ValueError("No frames were extracted from the video.")

        self.embeddings = np.asarray(embeddings, dtype=np.float32)

        elapsed = time.time() - start_time
        memory_mb = float(self.embeddings.nbytes / (1024 * 1024))

        self.stats["index_time_seconds"] = elapsed
        self.stats["total_frames"] = total_frame_count
        self.stats["sampled_frames"] = len(self.frames)
        self.stats["fps"] = len(self.frames) / elapsed if elapsed > 0 else 0.0
        self.stats["memory_mb"] = memory_mb
        self.stats["video_duration_seconds"] = duration_seconds
        self.stats["sampling_interval_seconds"] = frame_interval / fps

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search for scenes matching the natural language query.

        Args:
            query: Natural language description.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult objects sorted by confidence descending.

        Raises:
            RuntimeError: If load_video() hasn't been called yet.
            ValueError: If query is empty.
        """
        if self.video_path is None or self.embeddings is None:
            raise RuntimeError("Must call load_video() before search().")

        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        top_k = max(1, min(top_k, len(self.timestamps)))

        with torch.no_grad():
            text_tokens = open_clip.tokenize([query]).to(self.device)
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        text_embedding = text_embedding.cpu().numpy()[0].astype(np.float32)

        # Cosine similarity because embeddings are normalized
        similarities = self.embeddings @ text_embedding

        # Light temporal smoothing to reduce noisy peaks
        if len(similarities) >= 3:
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            smoothed = np.convolve(similarities, kernel, mode="same")
        else:
            smoothed = similarities

        # Pick best frames, skipping near-duplicate timestamps
        ranked_indices = np.argsort(smoothed)[::-1]
        selected_indices = []
        min_gap_seconds = 2.0

        for idx in ranked_indices:
            ts = self.timestamps[idx]
            too_close = any(abs(ts - self.timestamps[chosen]) < min_gap_seconds for chosen in selected_indices)
            if not too_close:
                selected_indices.append(int(idx))
            if len(selected_indices) >= top_k:
                break

        # Normalize confidence scores into a more readable range
        min_score = float(np.min(smoothed))
        max_score = float(np.max(smoothed))
        score_range = max(max_score - min_score, 1e-6)

        results: List[SearchResult] = []
        for idx in selected_indices:
            norm_conf = float((smoothed[idx] - min_score) / score_range)

            results.append(
                SearchResult(
                    timestamp_ms=int(self.timestamps[idx] * 1000),
                    confidence=norm_conf,
                    frame_number=int(self.sampled_frame_numbers[idx]),
                    thumbnail_path=None,
                )
            )

        return results

    def get_processing_stats(self) -> dict:
        """Return statistics about processing performance."""
        return self.stats"""
Video Search Implementation

This is the main entry point for your solution. You must implement the
VideoSearch class that inherits from VideoSearchInterface.

See the API documentation in docs/API_CONTRACT.md for details.
"""

from typing import List
import sys
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import open_clip
from PIL import Image

# Add parent directory to path for evaluation imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.interface import VideoSearchInterface, SearchResult


class VideoSearch(VideoSearchInterface):
    """
    CLIP-based video search over sampled frames.

    Approach:
    - Sample frames at a fixed interval
    - Encode sampled frames with CLIP
    - Encode text query with CLIP
    - Rank sampled frames by cosine similarity
    - Apply light temporal smoothing to improve stability
    """

    def __init__(self):
        """Initialize the video search system."""
        self.video_path = None
        self.frames = []
        self.timestamps = []
        self.embeddings = None
        self.sampled_frame_numbers = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "ViT-B-32"
        self.pretrained_name = "openai"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained_name,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.stats = {
            "fps": 0.0,
            "memory_mb": 0.0,
            "model_info": {
                "name": f"CLIP {self.model_name}",
                "pretrained": self.pretrained_name,
                "device": self.device,
            },
            "index_time_seconds": 0.0,
            "total_frames": 0,
            "sampled_frames": 0,
            "video_duration_seconds": 0.0,
            "sampling_interval_seconds": 1.0,
        }

    def load_video(self, video_path: str) -> None:
        """
        Load and preprocess a video.

        Args:
            video_path: Path to the video file.

        Raises:
            FileNotFoundError: If video doesn't exist.
            ValueError: If video format is not supported or video cannot be read.
        """
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        supported_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        if video_file.suffix.lower() not in supported_suffixes:
            raise ValueError(f"Unsupported video format: {video_file.suffix}")

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        self.video_path = str(video_file)
        self.frames = []
        self.timestamps = []
        self.sampled_frame_numbers = []
        self.embeddings = None

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = raw_fps if raw_fps and raw_fps > 0 else 25.0

        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = (
            total_frame_count / fps if total_frame_count > 0 and fps > 0 else 0.0
        )

        # Sample 1 frame per second for a good speed/accuracy balance
        frame_interval = max(1, int(round(fps)))

        start_time = time.time()
        embeddings = []
        frame_idx = 0

        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    timestamp = frame_idx / fps

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

                    image_embedding = self.model.encode_image(image_tensor)
                    image_embedding = image_embedding / image_embedding.norm(
                        dim=-1, keepdim=True
                    )

                    embeddings.append(image_embedding.cpu().numpy()[0])
                    self.frames.append(frame.copy())
                    self.timestamps.append(timestamp)
                    self.sampled_frame_numbers.append(frame_idx)

                frame_idx += 1

        cap.release()

        if not embeddings:
            raise ValueError("No frames were extracted from the video.")

        self.embeddings = np.asarray(embeddings, dtype=np.float32)

        elapsed = time.time() - start_time
        memory_mb = float(self.embeddings.nbytes / (1024 * 1024))

        self.stats["index_time_seconds"] = elapsed
        self.stats["total_frames"] = total_frame_count
        self.stats["sampled_frames"] = len(self.frames)
        self.stats["fps"] = len(self.frames) / elapsed if elapsed > 0 else 0.0
        self.stats["memory_mb"] = memory_mb
        self.stats["video_duration_seconds"] = duration_seconds
        self.stats["sampling_interval_seconds"] = frame_interval / fps

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search for scenes matching the natural language query.

        Args:
            query: Natural language description.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult objects sorted by confidence descending.

        Raises:
            RuntimeError: If load_video() hasn't been called yet.
            ValueError: If query is empty.
        """
        if self.video_path is None or self.embeddings is None:
            raise RuntimeError("Must call load_video() before search().")

        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        top_k = max(1, min(top_k, len(self.timestamps)))

        with torch.no_grad():
            text_tokens = open_clip.tokenize([query]).to(self.device)
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        text_embedding = text_embedding.cpu().numpy()[0].astype(np.float32)

        # Cosine similarity because embeddings are normalized
        similarities = self.embeddings @ text_embedding

        # Light temporal smoothing to reduce noisy peaks
        if len(similarities) >= 3:
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            smoothed = np.convolve(similarities, kernel, mode="same")
        else:
            smoothed = similarities

        # Pick best frames, skipping near-duplicate timestamps
        ranked_indices = np.argsort(smoothed)[::-1]
        selected_indices = []
        min_gap_seconds = 2.0

        for idx in ranked_indices:
            ts = self.timestamps[idx]
            too_close = any(abs(ts - self.timestamps[chosen]) < min_gap_seconds for chosen in selected_indices)
            if not too_close:
                selected_indices.append(int(idx))
            if len(selected_indices) >= top_k:
                break

        # Normalize confidence scores into a more readable range
        min_score = float(np.min(smoothed))
        max_score = float(np.max(smoothed))
        score_range = max(max_score - min_score, 1e-6)

        results: List[SearchResult] = []
        for idx in selected_indices:
            norm_conf = float((smoothed[idx] - min_score) / score_range)

            results.append(
                SearchResult(
                    timestamp_ms=int(self.timestamps[idx] * 1000),
                    confidence=norm_conf,
                    frame_number=int(self.sampled_frame_numbers[idx]),
                    thumbnail_path=None,
                )
            )

        return results

    def get_processing_stats(self) -> dict:
        """Return statistics about processing performance."""
        return self.stats

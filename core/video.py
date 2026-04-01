import av
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image
from deepface import DeepFace

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"


def get_face_encodings_from_image(image_path: str) -> list[np.ndarray]:
    """Extract face embeddings from an image using DeepFace."""
    try:
        img = np.array(Image.open(image_path).convert("RGB"))
        results = DeepFace.represent(
            img_path=img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
        return [np.array(r["embedding"]) for r in results if r["face_confidence"] > 0.6]
    except Exception as e:
        print(f"  [WARN] Image error {image_path}: {e}")
        return []


def get_face_encodings_from_video(
    video_path: str, sample_rate: int = 30, debug_dir: str | None = None
) -> list[np.ndarray]:
    """Sample frames from a video and extract face embeddings using DeepFace."""
    encodings = []
    frame_count = 0

    try:
        container = av.open(video_path)
    except Exception:
        return []

    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        p = Path(video_path)
        parent_hash = hashlib.md5(str(p.parent).encode()).hexdigest()[:6]
        stem = f"{parent_hash}_{p.stem}"

    for frame in container.decode(video=0):
        if frame_count % sample_rate == 0:
            try:
                img = frame.to_ndarray(format="bgr24")

                if debug_dir:
                    Image.fromarray(img[..., ::-1]).save(debug_path / f"{stem}_f{frame_count}.jpg")

                results = DeepFace.represent(
                    img_path=img,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                )
                found = [np.array(r["embedding"]) for r in results if r["face_confidence"] > 0.6]
                encodings.extend(found)
            except Exception:
                pass
        frame_count += 1

    container.close()
    return encodings


def get_face_encodings_from_folder(
    folder_path: str, sample_rate: int = 30, debug_dir: str | None = None
) -> list[np.ndarray]:
    """Scan a folder recursively and extract face encodings from all supported files."""
    folder = Path(folder_path)
    all_encodings: list[np.ndarray] = []

    for file in folder.rglob("*"):
        ext = file.suffix.lower()
        try:
            if ext in SUPPORTED_IMAGE_EXTS:
                all_encodings.extend(get_face_encodings_from_image(str(file)))
            elif ext in SUPPORTED_VIDEO_EXTS:
                all_encodings.extend(get_face_encodings_from_video(str(file), sample_rate, debug_dir))
        except Exception as e:
            print(f"  [WARN] Failed to process {file}: {e}")

    return all_encodings

import av
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}

FACE_CONFIDENCE_THRESHOLD = 0.6

# Choose your device: "cpu", "cuda" (NVIDIA GPU), or "mps" (Apple Silicon)
_device = torch.device("cpu")
_mtcnn = MTCNN(keep_all=True, device=_device)
_resnet = InceptionResnetV1(pretrained="vggface2").eval().to(_device)


def _extract_embeddings(img: Image.Image) -> list[np.ndarray]:
    """Extract face embeddings from a PIL RGB image."""
    faces, probs = _mtcnn(img, return_prob=True)
    if faces is None:
        return []
    embeddings = []
    with torch.no_grad():
        for face, prob in zip(faces, probs):
            if prob < FACE_CONFIDENCE_THRESHOLD:
                continue
            emb = _resnet(face.unsqueeze(0).to(_device))
            embeddings.append(emb.squeeze().cpu().numpy())
    return embeddings


def get_face_encodings_from_image(image_path: str) -> list[np.ndarray]:
    """Extract face embeddings from an image."""
    try:
        img = Image.open(image_path).convert("RGB")
        return _extract_embeddings(img)
    except Exception as e:
        print(f"  [WARN] Image error {image_path}: {e}")
        return []


def get_face_encodings_from_video(
    video_path: str, sample_rate: int = 30, debug_dir: str | None = None
) -> list[np.ndarray]:
    """Sample frames from a video and extract face embeddings."""
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
                img = frame.to_image().convert("RGB")

                if debug_dir:
                    img.save(debug_path / f"{stem}_f{frame_count}.jpg")

                encodings.extend(_extract_embeddings(img))
            except Exception:
                pass
        frame_count += 1

    container.close()
    return encodings


def get_face_encodings_from_folder(
    folder_path: str, sample_rate: int = 30, debug_dir: str | None = None
) -> list[np.ndarray]:
    """Scan a folder recursively. Images take priority; videos are used only if no faces found in images."""
    folder = Path(folder_path)
    encodings: list[np.ndarray] = []

    for file in folder.rglob("*"):
        if file.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            try:
                encodings.extend(get_face_encodings_from_image(str(file)))
            except Exception as e:
                print(f"  [WARN] Failed to process {file}: {e}")

    if encodings:
        return encodings

    for file in folder.rglob("*"):
        if file.suffix.lower() in SUPPORTED_VIDEO_EXTS:
            try:
                encodings.extend(get_face_encodings_from_video(str(file), sample_rate, debug_dir))
            except Exception as e:
                print(f"  [WARN] Failed to process {file}: {e}")

    return encodings

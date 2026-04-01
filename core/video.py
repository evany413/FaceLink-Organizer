import cv2
import numpy as np
from pathlib import Path
from deepface import DeepFace

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"


def get_face_encodings_from_image(image_path: str) -> list[np.ndarray]:
    """Extract face embeddings from an image using DeepFace."""
    try:
        results = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
        return [np.array(r["embedding"]) for r in results if r["face_confidence"] > 0.6]
    except Exception as e:
        print(f"  [WARN] Image error {image_path}: {e}")
        return []


def get_face_encodings_from_video(video_path: str, sample_rate: int = 30) -> list[np.ndarray]:
    """Sample frames from a video and extract face embeddings using DeepFace."""
    cap = cv2.VideoCapture(video_path)
    sar_num = cap.get(cv2.CAP_PROP_SAR_NUM)
    sar_den = cap.get(cv2.CAP_PROP_SAR_DEN)
    apply_sar = sar_num > 0 and sar_den > 0 and sar_num != sar_den
    encodings = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if apply_sar:
            new_w = int(frame.shape[1] * sar_num / sar_den)
            frame = cv2.resize(frame, (new_w, frame.shape[0]))

        if frame_count % sample_rate == 0:
            try:
                results = DeepFace.represent(
                    img_path=frame,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                )
                found = [np.array(r["embedding"]) for r in results if r["face_confidence"] > 0.6]
                encodings.extend(found)
            except Exception:
                pass

        frame_count += 1

    cap.release()
    return encodings


def get_face_encodings_from_folder(folder_path: str, sample_rate: int = 30) -> list[np.ndarray]:
    """Scan a folder recursively and extract face encodings from all supported files."""
    folder = Path(folder_path)
    all_encodings: list[np.ndarray] = []

    for file in folder.rglob("*"):
        ext = file.suffix.lower()
        try:
            if ext in SUPPORTED_IMAGE_EXTS:
                all_encodings.extend(get_face_encodings_from_image(str(file)))
            elif ext in SUPPORTED_VIDEO_EXTS:
                all_encodings.extend(get_face_encodings_from_video(str(file), sample_rate))
        except Exception as e:
            print(f"  [WARN] Failed to process {file}: {e}")

    return all_encodings

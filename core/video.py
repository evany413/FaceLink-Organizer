import cv2
import face_recognition
import numpy as np
from pathlib import Path

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}


def get_face_encodings_from_image(image_path: str) -> list[np.ndarray]:
    """Load an image and return all face encodings found in it."""
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image)
    return face_recognition.face_encodings(image, locations)


def get_face_encodings_from_video(video_path: str, sample_rate: int = 30) -> list[np.ndarray]:
    """Sample frames from a video and return all face encodings found."""
    cap = cv2.VideoCapture(video_path)
    encodings = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            found = face_recognition.face_encodings(rgb, locations)
            encodings.extend(found)

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

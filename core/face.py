import json
import numpy as np
from pathlib import Path


def encodings_to_json(encodings: list[np.ndarray]) -> list[list[float]]:
    return [e.tolist() for e in encodings]


def encodings_from_json(data: list[list[float]]) -> list[np.ndarray]:
    return [np.array(e) for e in data]


def save_cache(cache: dict[str, list[np.ndarray]], cache_path: str) -> None:
    """Persist encoding cache to a JSON file."""
    serializable = {k: encodings_to_json(v) for k, v in cache.items()}
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)


def load_cache(cache_path: str) -> dict[str, list[np.ndarray]]:
    """Load encoding cache from a JSON file. Returns empty dict if file missing."""
    path = Path(cache_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: encodings_from_json(v) for k, v in raw.items()}


def get_representative_encodings(
    encodings: list[np.ndarray], tolerance: float = 0.5
) -> list[np.ndarray]:
    """Return all encodings belonging to the most frequently appearing face cluster."""
    if not encodings:
        return []
    arr = np.array(encodings)
    best_cluster: np.ndarray = arr[:0]
    for enc in encodings:
        distances = np.linalg.norm(arr - enc, axis=1)
        cluster = arr[distances <= tolerance]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    return list(best_cluster)


def folders_share_face(
    encodings_a: list[np.ndarray],
    encodings_b: list[np.ndarray],
    tolerance: float = 0.5,
) -> bool:
    """Return True if any face in encodings_a matches any face in encodings_b."""
    if not encodings_a or not encodings_b:
        return False

    for enc_a in encodings_a:
        distances = np.linalg.norm(np.array(encodings_b) - enc_a, axis=1)
        if np.any(distances <= tolerance):
            return True
    return False

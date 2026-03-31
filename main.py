"""
FaceLink Organizer
==================
Scans a directory of sub-folders, groups them by shared faces using face
recognition + connected-components, then moves each group into a numbered
top-level folder (001/, 002/, …).

Usage:
    uv run main.py <target_dir> [options]

Options:
    --tolerance FLOAT   Face-match distance threshold (default: 0.5)
    --sample-rate INT   Video frame sampling interval (default: 30)
    --cache PATH        Path to encoding cache JSON (default: cache/encodings.json)
    --dry-run           Print what would happen without moving anything
"""

import argparse
import shutil
from pathlib import Path

import networkx as nx

from core.face import folders_share_face, load_cache, save_cache
from core.video import get_face_encodings_from_folder


# ---------------------------------------------------------------------------
# Step 1 – Scan & extract encodings (with cache)
# ---------------------------------------------------------------------------

def build_encoding_map(
    folders: list[Path],
    cache_path: str,
    sample_rate: int,
) -> dict[str, list]:
    """Return {folder_str: [encodings]} for every sub-folder, using cache."""
    cache = load_cache(cache_path)
    changed = False

    for folder in folders:
        key = str(folder)
        if key in cache:
            print(f"  [cache] {folder.name}")
            continue

        print(f"  [scan]  {folder.name} … ", end="", flush=True)
        encodings = get_face_encodings_from_folder(key, sample_rate)
        print(f"{len(encodings)} face(s) found")
        cache[key] = encodings
        changed = True

    if changed:
        save_cache(cache, cache_path)
        print(f"  Cache saved → {cache_path}\n")

    return cache


# ---------------------------------------------------------------------------
# Step 2 – Build graph & find connected components
# ---------------------------------------------------------------------------

def build_groups(
    encoding_map: dict[str, list],
    tolerance: float,
) -> list[set[str]]:
    """Compare every pair of folders and cluster via connected components."""
    folders = list(encoding_map.keys())
    G = nx.Graph()
    G.add_nodes_from(folders)

    for i in range(len(folders)):
        for j in range(i + 1, len(folders)):
            a, b = folders[i], folders[j]
            if folders_share_face(encoding_map[a], encoding_map[b], tolerance):
                G.add_edge(a, b)
                print(f"  [match] {Path(a).name}  <->  {Path(b).name}")

    return list(nx.connected_components(G))


# ---------------------------------------------------------------------------
# Step 3 – Move folders into numbered group directories
# ---------------------------------------------------------------------------

def organise(
    target_dir: Path,
    groups: list[set[str]],
    dry_run: bool,
) -> None:
    """Move each sub-folder into its numbered group directory."""
    # Sort groups so the largest group gets 001
    groups_sorted = sorted(groups, key=len, reverse=True)

    for idx, group in enumerate(groups_sorted, start=1):
        group_dir = target_dir / f"{idx:03d}"
        print(f"\nGroup {idx:03d}/ ({len(group)} folder(s)):")

        for folder_str in sorted(group):
            src = Path(folder_str)
            dst = group_dir / src.name
            print(f"  {src.name}  →  {dst.relative_to(target_dir)}")
            if not dry_run:
                group_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

    if dry_run:
        print("\n[dry-run] No files were moved.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Group folders by shared faces.")
    parser.add_argument("target_dir", help="Directory containing the sub-folders to organise")
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="Face-match distance threshold (default: 0.5)")
    parser.add_argument("--sample-rate", type=int, default=30,
                        help="Video frame sampling interval (default: 30)")
    parser.add_argument("--cache", default="cache/encodings.json",
                        help="Path to encoding cache JSON (default: cache/encodings.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview moves without executing them")
    args = parser.parse_args()

    target_dir = Path(args.target_dir).resolve()
    if not target_dir.is_dir():
        raise SystemExit(f"Error: '{target_dir}' is not a directory.")

    # Collect immediate sub-folders only (top-level children)
    folders = sorted([p for p in target_dir.iterdir() if p.is_dir()])
    if not folders:
        raise SystemExit("No sub-folders found in the target directory.")

    print(f"Target : {target_dir}")
    print(f"Folders: {len(folders)} found\n")

    # 1. Extract / load encodings
    print("=== Step 1: Extracting face encodings ===")
    encoding_map = build_encoding_map(folders, args.cache, args.sample_rate)

    # 2. Build graph & cluster
    print("=== Step 2: Building association graph ===")
    groups = build_groups(encoding_map, args.tolerance)

    # 3. Move folders
    print("\n=== Step 3: Organising folders ===")
    organise(target_dir, groups, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()

# FaceLink Organizer

An intelligent file management tool that groups folders based on facial recognition and graph-based connectivity. It scans images and videos within subdirectories, identifies common individuals, and clusters related folders into organized top-level directories.

## Features

- **Facial Feature Extraction** — Detects and encodes faces from images and video frames.
- **Relational Clustering** — Uses Connected Components to link folders transitively (e.g. if Folder A and B share Person 1, and B and C share Person 2, all three end up in the same group).
- **Structure-Preserving** — Moves entire folder structures into numbered parent directories (`001/`, `002/`, …) without altering internal content.
- **Caching** — Face encodings are persisted to a local JSON cache to avoid redundant computation on re-runs.
- **Video Support** — Frame-skipping keeps video processing fast (configurable sample rate).

## Getting Started

### Prerequisites

Install [uv](https://docs.astral.sh/uv/):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> **Note:** The first run will download model weights (`vggface2` via facenet-pytorch) automatically.

### Installation

**Mac (Apple Silicon / Intel) or CPU-only:**

Before running `uv sync`, comment out the two CUDA blocks at the bottom of `pyproject.toml`:

```toml
# [[tool.uv.index]]
# name = "pytorch-cu121"
# url = "https://download.pytorch.org/whl/cu121"
# explicit = true

# [tool.uv.sources]
# torch = { index = "pytorch-cu121" }
# torchvision = { index = "pytorch-cu121" }
```

Then:

```bash
git clone https://github.com/evany413/FaceLink-Organizer.git
cd FaceLink-Organizer
uv sync
```

On Apple Silicon, PyTorch will automatically use the **MPS** backend for GPU acceleration — no extra steps needed.

**Windows / Linux with NVIDIA GPU:**

1. Install the latest [NVIDIA driver](https://www.nvidia.com/drivers)
2. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
3. In `pyproject.toml`, update the CUDA version in the index URL to match your installation (e.g. `cu118`, `cu121`, `cu124`) and leave the two CUDA blocks uncommented
4. Run:

```bash
git clone https://github.com/evany413/FaceLink-Organizer.git
cd FaceLink-Organizer
uv sync
```

## Usage

```bash
# Preview results without moving anything
uv run main.py /path/to/folders --dry-run

# Run the organizer
uv run main.py /path/to/folders
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--tolerance` | `0.5` | Face-match distance threshold (lower = stricter) |
| `--sample-rate` | `30` | Video frame sampling interval |
| `--cache` | `cache/encodings.json` | Path to encoding cache file |
| `--dry-run` | — | Preview moves without executing them |
| `--debug [DIR]` | `debug_frames/` | Save sampled video frames for inspection |

## How It Works

1. **Scan** — Collects all immediate subdirectories of the target path.
2. **Extract** — Processes images and sampled video frames to generate 512-d face embeddings (InceptionResnetV1 / vggface2), saving results to a local cache.
3. **Match** — Computes Euclidean distance between encodings; a distance ≤ `tolerance` is considered a match.
4. **Graph** — Each folder becomes a node; an edge is drawn between two nodes if they share at least one face.
5. **Cluster** — `nx.connected_components()` finds all groups of directly or indirectly linked folders.
6. **Organize** — Creates `001/`, `002/`, … directories and moves each group's folders into them.

## Tech Stack

| Role | Library |
|---|---|
| Face detection & embedding | `facenet-pytorch` (MTCNN + InceptionResnetV1) |
| GPU acceleration | PyTorch — CUDA / MPS / CPU auto-detected |
| Video decoding | `av` (PyAV / FFmpeg) |
| Image loading | `Pillow` |
| Graph clustering | `networkx` |
| Environment | `uv` |

## Project Structure

```
FaceLink-Organizer/
├── .python-version
├── pyproject.toml
├── main.py               # CLI entry point
├── core/
│   ├── video.py          # Image & video processing
│   └── face.py           # Encoding comparison & cache I/O
└── cache/                # Persisted face encodings (JSON)
```

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

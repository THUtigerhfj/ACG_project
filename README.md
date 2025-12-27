# ACG Real-Time SPH Project

GPU-accelerated, real-time fluid simulation using Position-Based Fluids (PBF) implemented in Python with NVIDIA Warp. Move a kinematic container with keyboard controls and watch the water respond with incompressible fluid dynamics and XSPH viscosity damping.

## Features

- **Real-time PBF simulation** with GPU acceleration via NVIDIA Warp
- **Interactive container control** using keyboard inputs
- **XSPH viscosity** for smooth, realistic fluid behavior
- **SDF-based collision** handling for container boundaries
- **Two-way coupled rigid spheres**: one lighter and one heavier than water, interacting with the fluid via PBD-style contact impulses
- **PyVista visualization** with ~60 FPS rendering

## Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x

### Installation

1. Create/activate a Python 3.10+ environment:

   ```bash
   conda create -n sph python=3.10
   conda activate sph
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the smoothed surface viewer with rigid spheres enabled:

  ```bash
  python -m src.app.smoothed_viewer
  ```

## Controls

### Container Movement

| Key | Action |
|-----|--------|
| **A** | Move container left (-X) |
| **D** | Move container right (+X) |
| **X** | Move container up (+Y) |
| **S** | Move container down (-Y) |
| **Q** | Move container backward (-Z) |
| **E** | Move container forward (+Z) |
| **R** | Reset container to origin |

### Camera Controls (Mouse)

| Input | Action |
|-------|--------|
| **Left-drag** | Rotate camera |
| **Right-drag** | Pan camera |
| **Scroll** | Zoom in/out |

## Command Line Options

```bash
python -m src.app.realtime_viewer [OPTIONS]

Options:
  --config PATH    Path to YAML config file (default: configs/default.yaml)
  --device DEVICE  Target device: cuda or cpu (default: cuda)
  --frames N       Optional frame limit for testing
```

## Configuration

Edit `configs/default.yaml` to adjust simulation parameters:

## Useful Scripts

- `python scripts/generate_initial_state.py` - Generate particle initial states
- `python debug_test.py` - Step-by-step simulation debugging
- `pytest` - Run unit tests

## Repository Layout

See `docs/project_overview.md` for detailed architecture documentation.

```text
ACG_project/
├── configs/          # Simulation presets (YAML)
├── docs/             # Documentation
├── src/
│   ├── app/          # Viewer and entry points
│   ├── kernels/      # Warp GPU kernels
│   ├── sim/          # Simulation components
│   └── utils/        # Configuration utilities
├── tests/            # Unit tests
└── requirements.txt  # Dependencies
```

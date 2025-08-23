# Periodic Error Analysis of Equatorial Mount with Optical Flow

[![Português](https://img.shields.io/badge/Português-README-green.svg)](README.md)

This project uses computer vision techniques, specifically optical flow, to measure and analyze with high precision the periodic error and other motion anomalies in the Right Ascension (RA) axis of a telescope mount.

The system was developed to characterize a Meade LXD55 mount, using a USB microscope to capture images of a surface attached to the RA axis. Motion data is processed in real-time to provide a detailed analysis of the mount's performance in physical units (arcseconds).

## Features

* **Displacement Measurement:** Uses the state-of-the-art optical flow algorithm **RAFT** (via PyTorch) with GPU acceleration (CUDA) for sub-pixel measurements.
* **Real-Time Analysis:** Calculates velocity, position, and applies statistical filters for outlier rejection.
* **Advanced Frequency Analysis:** Applies a signal processing chain (Detrending, Hanning Window) to generate a clean Frequency Spectrum (FFT), identifying the components of the periodic error.
* **Dynamic Visualization:** Presents data in a dashboard with multiple charts (Velocity vs. Time, FFT, etc.) using Matplotlib.
* **Physical Calibration:** Converts measurements from `pixels/s` to `arcseconds/s` through automatic calibration based on the theoretical speed of the mount.
* **Session Management:** Automatically saves all raw data (images and velocity CSV) and metadata (analysis parameters) in unique session folders for each run.
* **Offline Analysis:** Includes a script to load, reprocess, and analyze previously saved data sessions.

## Project Structure

The code has been refactored into a modular architecture for greater clarity and maintainability, located in `src/raft/`.

* **`config.py`**: Central file for all constants and configuration parameters, such as file paths, physical mount parameters, and analysis thresholds.
* **`session_manager.py`**: Module responsible for creating and managing capture sessions. Handles folder creation, saving metadata (`metadata.json`), velocity data (`data.csv`), and images.
* **`analysis.py`**: The "brain" of the analysis. Contains all numerical processing logic (moving averages, FFT, peak detection, etc.). Has no visualization code.
* **`visualization.py`**: Responsible for all visual output. Creates and updates Matplotlib charts and draws the information frame (HUD) on the OpenCV video.
* **`main_live.py`**: Entry point for **data acquisition** (either from the live camera or reprocessing an image folder). Orchestrates the other modules to capture, analyze, visualize, and save data in real-time.
* **`main_offline.py`**: Entry point for **analysis of already saved data**. Loads a `data.csv` file from a previous session and generates the final charts.

## Installation and Execution

### 1. Requirements

The project uses Python 3.11+. Dependencies are listed in the `requirements.txt` file.

### 2. Environment Setup

It is highly recommended to use a virtual environment.

```bash
# 1. Navigate to the project root folder (D:\RAFT)
cd D:\RAFT

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate
```

### 3. Installing Dependencies

With the environment activated, install the necessary packages:

```bash
pip install -r requirements.txt
```
**Note:** The `requirements.txt` file is configured to install the PyTorch version with **CUDA 12.1** support. Make sure your NVIDIA drivers are compatible.

### 4. Execution

All scripts should be executed as modules from the project root folder (`D:\RAFT`).

* **For Real-Time Analysis (Camera or Image Folder):**
    * Configure the desired parameters in `src/raft/config.py` (`USE_LIVE_CAMERA`, `IMAGE_FOLDER_PATH`, etc.).
    * Run the command:
    ```bash
    python -m src.raft.main_live
    ```

* **For Analysis of a Saved CSV:**
    * Configure the file path in `src/raft/config.py` (variable `SESSION_CSV_PATH`).
    * Run the command:
    ```bash
    python -m src.raft.main_offline
    ```

# tanalysis

A Python library for processing and analyzing cell trajectories in 2D and 3D time-lapse videos.

## Overview

**tanalysis** provides a complete workflow for analyzing cell dynamics from microscopy videos. The library integrates cell detection (via implemented algorithm or [Cellpose](https://cellpose.readthedocs.io/)), simple image stitching for tiles reconstruction, cell tracking (via implemented LAP tracker or [TrackMate](https://imagej.net/plugins/trackmate/)), and trajectory analysis.

Recommended workflow is implemented in trajectory_analysis.ipynb.

### Capabilities

- **Image I/O**: Support for `.tif`, `.tiff`, and `.lif` (Leica) microscopy formats.
- **Cell detection**: Simple spot detector for clear videos. For more complex videos, [Cellpose](https://cellpose.org/) should be used.
- **Image Stitching**: Alignment and stitching of tiled microscopy images using peak correlation matching.
- **Cell Tracking**: Centroid-based tracking using linear assignment problem (LAP).
- **Trajectory Analysis**: Compute motion metrics (velocity, displacement, spatial coverage, etc.) and statistical comparisons (Cohen's d value).

## Installation

### Requirements

- Python 3.12
- Poetry (for dependency management)

### Setup

1. Clone this repository

2. Install dependencies using poetry or with pip using the requirements.txt file

### Usage

Basic workflow is described in trajectory_analysis.ipynb.

## Notes

· All images in a batch must have consistent dimensions.
· GPU acceleration significantly improves performance for large datasets

## Authors

Pau Canaleta Vicente - mailto:pcanaleta@ibecbarcelona.eu

## Citation

If you use tanalysis in your research, please cite this repository.

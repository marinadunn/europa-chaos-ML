# Detection and Segmentation of Ice Blocks in Europa's Chaos Terrain Using Mask R-CNN
Automated instance segmentation for ice blocks within Europa's chaos terrain using corrected NASA Galileo images.


## Data
The project utilizes imaging data from regional image mosaic maps ("RegMaps") produced by NASA's Galileo Solid State Imager (SSI) and photogrammetrically corrected by the [USGS Astrogeology Science Center](https://astrogeology.usgs.gov/search/map/Europa/Mosaic/Equirectangular_Mosaics_of_Europa_v3) as part of [Bland et al. 2021](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021EA001935).

The chaos regions use the following corresponding RegMaps for their labels:
- Chaos A, B, C, D, E: `15ESREGMAP02` (Leading Hemisphere)
- Chaos F, G, H, I: `17ESREGMAP02` (Leading Hemisphere)
- Chaos aa, bb: `17ESREGMAP01`, `17ESNERTRM01`, `11ESREGMAP01` (Trailing Hemisphere)
- Chaos Co: `E6ESDRKLIN01` (Trailing Hemisphere)
- Chaos dd: `11ESREGMAP01` (Trailing Hemisphere)
- Chaos ee, hh, ii, jj, kk: `17ESNERTRM01` (Trailing Hemisphere)
- Chaos ff, gg: `17ESREGMAP01` (Trailing Hemisphere)

## Installation / Requirements
To clone the repository and set up the environment:

1. Clone repository:
  ```
  git clone https://github.com/marinadunn/europa-chaos-ML.git
  cd europa-chaos-ML
  ```

2. Set up a new Python virtual environment
  It is recommended to create a new Python environment before installing packages and running the pipeline. All necessary package dependencies can be found in the `deps` directory.

  If running on a machine with CUDA GPU capability, run the following commands in a terminal:
  ```
  conda env create --name [envname]
  conda activate [envname]
  pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
  pip3 install -r deps/requirements.txt
  python3 -m ipykernel install --user --name [envname]
  ```

  Otherwise, for a CPU-only device, run the following:
  ```
  conda create --name [envname] python=3.10 gdal -c conda-forge
  conda activate [envname]
  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
  pip3 install -r deps/requirements.txt
  python3 -m ipykernel install --user --name [envname]
  ```

## Code Authors
- Marina Dunn (<mdunn014@ucr.edu>)
- Ahmed Awadallah (<ahmed.d8k8@gmail.com>)

Supervised by: Dr. Conor A. Nixon (<conor.a.nixon@nasa.gov>)

## References
If you use this code, please cite our paper: (link TBD)

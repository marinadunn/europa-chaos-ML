# Detection and Segmentation of Ice Blocks in Europa's Chaos Terrain Using Mask R-CNN

Automated instance segmentation for ice blocks within Europa's chaos terrain using corrected NASA Galileo images.

## Data

The project utilizes imaging data from regional image mosaic maps ("RegMaps") produced by NASA's Galileo Solid State Imager (SSI) and photogrammetrically corrected by the [USGS Astrogeology Science Center](https://astrogeology.usgs.gov/search/map/Europa/Mosaic/Equirectangular_Mosaics_of_Europa_v3) as part of [Bland et al. 2021](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021EA001935).

The labels used for model training are created by Alyssa C. Mills and are available publicly on [Zenodo](https://zenodo.org/records/10162452).

The data for each chaos region can be found in the `data` directory. Smaller image and mask tiles generated as part of the hyperparameter search process are stored in a new directory `processed_data`, and are sorted into training and testing subsets (`img_train`, `img_test`, `lbl_train`, `lbl_test`).

| Region | Mosaic ID | Resolution (m/pixel) | Leading/Trailing Hemisphere|
|--------|----------|----------|----------|
|   A    | 15ESREGMAP02                                 | 229     | Leading |
|   aa   | 17ESREGMAP01, 17ESNERTRM01, 11ESREGMAP01     | 210, 218, 222     | Trailing |
|   B    | 15ESREGMAP02                                 | 229     | Leading |
|   bb   | 17ESREGMAP01, 17ESNERTRM01, 11ESREGMAP01     | 210, 218, 222     | Trailing |
|   C    | 15ESREGMAP02                                 | 229     | Leading |
|   Co   | E6ESDRKLIN01                                 | 179     | Trailing |
|   D    | 15ESREGMAP02                                 | 229     | Leading |
|   dd   | 11ESREGMAP01                                 | 218     | Trailing |
|   E    | 15ESREGMAP02                                 | 229     | Leading |
|   ee   | 17ESNERTRM01                                 | 210     | Trailing |
|   F    | 17ESREGMAP02                                 | 215     | Leading |
|   ff   | 17ESREGMAP01                                 | 222     | Trailing |
|   G    | 17ESREGMAP02                                 | 215     | Leading |
|   gg   | 17ESREGMAP01                                 | 222     | Trailing |
|   H    | 17ESREGMAP02                                 | 215     | Leading |
|   hh   | 17ESNERTRM01                                 | 210     | Trailing |
|   I    | 17ESREGMAP02                                 | 215     | Leading |
|   ii   | 17ESNERTRM01                                 | 210     | Trailing |
|   jj   | 17ESNERTRM01                                 | 210     | Trailing |
|   kk   | 17ESNERTRM01                                 | 210     | Trailing |

## Model

Our current setup leverages a [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf) model framework, with a [ResNet50](https://arxiv.org/abs/1512.03385) backbone, and uses pre-trained weights from the [Common Objects in COntext (COCO)](https://cocodataset.org/#home) dataset to perform transfer learning.

Final trained models and their weights can be found in the `models` directory.

## Source Code

All source code can be found in the `src` directory.

## Examples

The directory `examples` includes the following:
- `reproject.py`: example script for reprojecting RegMap GeoTIFF images to the same CRS as the chaos GeoJSON files
- `pixel_metrics_example`: example script for making predictions and calculating pixel-level metrics for an image.
- `maskfromcoco.ipynb`: example notebook for how to create segmentation masks from COCO-formatted labels.

## Tests

All test code and resulting plots can be found in the `tests` directory.

## Hyperparameter Tuning

A hyperparameter search is conducted using the [`Optuna`](https://optuna.org) framework, and can be found in `src/optuna_main.py`. The search aims to maximize the segmentation IoU score across images in a dataset during optimization. The default Tree-structured Parzen Estimator (TPE), a form of Bayesian Optimization, is used as the sampler in order to perform a more efficient search.

To run a search, modify the hyperparameters within the selected objective function in `src/utils/optuna_utility/objectives.py` then run `python3 src/optuna_main.py -m [metric]`, where "metric" is either "f1", "precision", or "recall". The best results from this are saved in the `output/optuna_output` directory.

## Current Results

The files containing the best model results can be found in the `output` directory, and corresponding plots can be found in the `plots` directory.

## Installation / Requirements

Current branch: `main`

Project Repository Structure:
```
europa-chaos-ML/
├── data
│   ├── Chaos_[X]/
│       ├── image/
│       ├── label/
│       ├── geojson/
│   ├── processed_data/
│   ├── size_distributions/
├── deps/
├── examples/
├── models/
│   ├── model_weights/
├── plots/
├── src/
│   ├── model_objects/
│   ├── utils/
├── tests/
├── README.md
```

1. Clone Repository:
```
git clone https://github.com/marinadunn/europa-chaos-ML.git
cd europa-chaos-ML
```

2. Set Up a New Python Virtual Environment and Install Dependencies

It is recommended to create a new Python environment before installing packages and running the pipeline. All necessary dependencies can be found in the `deps` directory.

To run on a machine with CUDA GPU capability, use the run the following commands in a terminal:
```
conda create --name [envname]
conda activate [envname]
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r deps/requirements.txt
python3 -m ipykernel install --user --name [envname]
```

If running specifically on an internal NASA science-managed cloud environment, which has CUDA GPU capability, run the following commands in a terminal:
```
conda create --name [envname]
conda activate [envname]
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r deps/requirements_nasa.txt
python3 -m ipykernel install --user --name [envname]
```

3. (Optional) Perform Hyperparameter Search with Optuna

To run a hyperparameter search with optuna, run the following commands in a terminal to execute the Python script with the desired options:
```
cd src
python3 optuna_main.py [metric option]
```

Options include:
- `--metric` (str): Which metric to use for optuna hyperparameter optimization. Options currently include f1, precision, or recall. (Required)

4. Running Leave-One-Out Cross-Validation (LOOCV)

To perform the same LOOCV experiment setup currently described in our paper/poster, run the following commands in a terminal to execute the Python script:
```
cd src
python3 cross_val.py
```

## Authors

- **Marina M. Dunn (<marina.dunn@email.ucr.edu>)**
- Alyssa C. Mills (<alyssa_mills1@baylor.edu>)
- Ahmed Awadallah (<ahmed.d8k8@gmail.com>)
- Ethan J. Duncan (<ejduncan@asu.edu>)
- Douglas M. Trent (<douglas.m.trent@nasa.gov>)
- Andrew Larsen (<drewlarsen27@gmail.com>)
- John Santerre (<john.santerre@gmail.com>)
- Conor A. Nixon (<conor.a.nixon@nasa.gov>)

## References
If you use this code, please cite our [NeurIPS paper](https://ml4physicalsciences.github.io/2023/files/NeurIPS_ML4PS_2023_156.pdf).

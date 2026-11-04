Hello, welcome to the supplementary data of our paper "Predicting the Hydrostatic Limit of Pressure Transmitting Media" !

The data available under the directory "data" is specified there. Files found in the directories "notebooks" and "scripts" are useful for building the ML models (gen-1 and gen-2) - these were prepared by Gabriele C. Sosso (also responsible for building and using the ML for the predictions).
Here is a set of instructions that should help you use the gen-2 model:

1. Create an environment from `requirements.txt`.
2. Open `notebooks/01_ML_REPRODUCTION.ipynb`.
3. Consult `data/FIGURE_DATA_INDEX.csv` for the exact panel-to-file mapping.

`scripts/recompute_gen2_oof.py` reruns the final 10-times repeated 10-fold Extra-Trees analysis. `scripts/recompute_gen1_oof.py` reruns the leakage-controlled Gen-1 neural-network reconstruction and uses the separate Python-3.9 environment documented in `requirements-gen1.txt`. 

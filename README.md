## Start here

1. Create an environment from `requirements.txt`.
2. Open `notebooks/01_ML_REPRODUCTION.ipynb`.
3. Consult `data/FIGURE_DATA_INDEX.csv` for the exact panel-to-file mapping.

`scripts/recompute_gen2_oof.py` reruns the final 10-times repeated 10-fold Extra-Trees analysis. `scripts/recompute_gen1_oof.py` reruns the leakage-controlled Gen-1 neural-network reconstruction and uses the separate Python-3.9 environment documented in `requirements-gen1.txt`. 

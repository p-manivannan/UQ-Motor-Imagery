# To-Do:
- ensemble.py in models doesn't work. Refer to models_test.ipynb for details
- Clean up and make the following pipelines as modular as possible:
    - Tuning
    - Training
    - Predicting
    - Analysis

# Done:
- Pre-processing 👍
- Removed extra save/load dict to hdf5 functions
- Converted most of models_bachelors.py to classes that inherit from a base model. All of it works except ensembles
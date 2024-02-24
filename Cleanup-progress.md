# To-Do:
- Interface that provides hypermodel function to tuner based on method name
- dropout class in models doesn't work. Refer to models_test.ipynb for details
- Create config writing/loading to access method names
- Clean up and make the following pipelines as modular as possible:
    - Tuning
    - Training
    - Predicting
    - Analysis

# Done:
- Pre-processing 👍
- Removed extra save/load dict to hdf5 functions
- Converted most of models_bachelors.py to classes that inherit from a base model. All of it works except ensembles
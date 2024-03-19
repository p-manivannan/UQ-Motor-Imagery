# To-Do:
- Test and bug fix training pipeline
- Clean up and make the following pipelines as modular as possible:
    - Predicting
    - Analysis
- Refer to tune_methods.py for improvements to tuning
- Create config writing/loading to access method names:
    - As well as the training/tuning params for each method

# Done:
- Pre-processing 👍
- Removed extra save/load dict to hdf5 functions
- Converted most of models_bachelors.py to classes that inherit from a base model
- Tuning 👍
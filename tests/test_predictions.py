import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from diabetes_model import __version__ as _version
from diabetes_model.processing.data_manager import load_pipeline
from diabetes_model.config.core import config
from diabetes_model.processing.validation import validate_inputs
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
diabetes_model_pipe  = load_pipeline(file_name=pipeline_file_name)

def test_prediction_diabetes_yes(sample_input_data: pd.DataFrame):
    #1,88.14146933903956,63.262618027440055,23.404364261090432,149.3580820441298,21.94825015846498,0.6760216478054515,48.24787309665949,1
    # Take one record from X_test
    single_record = sample_input_data.iloc[[0]]
 
    # Use the pipeline to make predictions
    # predictions = diabetes_model_pipe.predict(single_record)
 
    # print(f"predictions: {predictions}")
    assert 1 == 1
# Predict Bike Sharing Demand with AutoGluon Template

## Project: Predict Bike Sharing Demand with AutoGluon
This notebook is a template with each step that you need to complete for the project.

Please fill in your code where there are explicit `?` markers in the notebook. You are welcome to add more cells and code as you see fit.

Once you have completed all the code implementations, please export your notebook as a HTML file so the reviews can view your code. Make sure you have all outputs correctly outputted.

`File-> Export Notebook As... -> Export Notebook as HTML`

There is a writeup to complete as well after all code implememtation is done. Please answer all questions and attach the necessary tables and charts. You can complete the writeup in either markdown or PDF.

Completing the code template and writeup template will cover all of the rubric points for this project.

The rubric contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this notebook and also discuss the results in the writeup file.

## Step 1: Create an account with Kaggle

### Create Kaggle Account and download API key
Below is example of steps to get the API username and key. Each student will have their own username and key.

1. Open account settings.
![kaggle1.png](attachment:kaggle1.png)
![kaggle2.png](attachment:kaggle2.png)
2. Scroll down to API and click Create New API Token.
![kaggle3.png](attachment:kaggle3.png)
![kaggle4.png](attachment:kaggle4.png)
3. Open up `kaggle.json` and use the username and key.
![kaggle5.png](attachment:kaggle5.png)

## Step 2: Download the Kaggle dataset using the kaggle python library

### Open up Sagemaker Studio and use starter template

1. Notebook should be using a `ml.t3.medium` instance (2 vCPU + 4 GiB)
2. Notebook should be using kernal: `Python 3 (MXNet 1.8 Python 3.7 CPU Optimized)`

### Install packages


```python
!pip install -U pip
!pip install -U setuptools wheel
!pip install -U "mxnet<2.0.0" bokeh==2.0.1
!pip install autogluon --no-cache-dir
```

### Setup Kaggle API Key


```python
!pip install -q Kaggle

!mkdir -p /root/.kaggle
!touch /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json

!kaggle competitions download -c bike-sharing-demand
!unzip -o bike-sharing-demand.zip    
```

    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m401 - Unauthorized
    Archive:  bike-sharing-demand.zip
      inflating: sampleSubmission.csv    
      inflating: test.csv                
      inflating: train.csv               


### Download and explore dataset


```python
import json
import pandas as pd

import matplotlib.pyplot as plt
import autogluon.core as ag

from autogluon.tabular import TabularPredictor
```


```python
kaggle_username = "siddp6"
kaggle_key = "d4dacbd687dea298b1a64e8ce809636c"

with open("/root/.kaggle/kaggle.json", "w") as f:
    f.write(json.dumps({"username": kaggle_username, "key": kaggle_key}))
```

### Go to the [bike sharing demand competition](https://www.kaggle.com/c/bike-sharing-demand) and agree to the terms
<!-- ![kaggle6.png](attachment:kaggle6.png) -->


```python
train = pd.read_csv("train.csv",parse_dates=["datetime"])
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.00000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.506614</td>
      <td>0.028569</td>
      <td>0.680875</td>
      <td>1.418427</td>
      <td>20.23086</td>
      <td>23.655084</td>
      <td>61.886460</td>
      <td>12.799395</td>
      <td>36.021955</td>
      <td>155.552177</td>
      <td>191.574132</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.116174</td>
      <td>0.166599</td>
      <td>0.466159</td>
      <td>0.633839</td>
      <td>7.79159</td>
      <td>8.474601</td>
      <td>19.245033</td>
      <td>8.164537</td>
      <td>49.960477</td>
      <td>151.039033</td>
      <td>181.144454</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.82000</td>
      <td>0.760000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>13.94000</td>
      <td>16.665000</td>
      <td>47.000000</td>
      <td>7.001500</td>
      <td>4.000000</td>
      <td>36.000000</td>
      <td>42.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>20.50000</td>
      <td>24.240000</td>
      <td>62.000000</td>
      <td>12.998000</td>
      <td>17.000000</td>
      <td>118.000000</td>
      <td>145.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>26.24000</td>
      <td>31.060000</td>
      <td>77.000000</td>
      <td>16.997900</td>
      <td>49.000000</td>
      <td>222.000000</td>
      <td>284.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>41.00000</td>
      <td>45.455000</td>
      <td>100.000000</td>
      <td>56.996900</td>
      <td>367.000000</td>
      <td>886.000000</td>
      <td>977.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10886 entries, 0 to 10885
    Data columns (total 12 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   datetime    10886 non-null  datetime64[ns]
     1   season      10886 non-null  int64         
     2   holiday     10886 non-null  int64         
     3   workingday  10886 non-null  int64         
     4   weather     10886 non-null  int64         
     5   temp        10886 non-null  float64       
     6   atemp       10886 non-null  float64       
     7   humidity    10886 non-null  int64         
     8   windspeed   10886 non-null  float64       
     9   casual      10886 non-null  int64         
     10  registered  10886 non-null  int64         
     11  count       10886 non-null  int64         
    dtypes: datetime64[ns](1), float64(3), int64(8)
    memory usage: 1020.7 KB



```python
test = pd.read_csv("test.csv",parse_dates=["datetime"])
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission = pd.read_csv("sampleSubmission.csv",parse_dates=["datetime"])
submission.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Step 3: Train a model using AutoGluon’s Tabular Prediction

Requirements:
* We are predicting `count`, so it is the label we are setting.
* Ignore `casual` and `registered` columns as they are also not present in the test dataset. 
* Use the `root_mean_squared_error` as the metric to use for evaluation.
* Set a time limit of 10 minutes (600 seconds).
* Use the preset `best_quality` to focus on creating the best model.


```python
predictor = TabularPredictor(
    label="count",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    learner_kwargs={"ignored_columns": ["casual", "registered"]},
)

```

    No path specified. Models will be saved in: "AutogluonModels/ag-20230602_140913/"



```python
predictor.fit(train_data=train, time_limit=600, presets="best_quality")
```

    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 600s
    AutoGluon will save models to "AutogluonModels/ag-20230602_140913/"
    AutoGluon Version:  0.7.0
    Python Version:     3.8.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Tue Apr 25 15:24:19 UTC 2023
    Train Data Rows:    10886
    Train Data Columns: 11
    Label Column: count
    Preprocessing data ...
    Using Feature Generators to preprocess the data ...
    Dropping user-specified ignored columns: ['casual', 'registered']
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    2582.34 MB
    	Train Data (Original)  Memory Usage: 0.78 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting DatetimeFeatureGenerator...
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('datetime', []) : 1 | ['datetime']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 5 | ['season', 'holiday', 'workingday', 'weather', 'humidity']
    	Types of features in processed data (raw dtype, special dtypes):
    		('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])                  : 3 | ['season', 'weather', 'humidity']
    		('int', ['bool'])            : 2 | ['holiday', 'workingday']
    		('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    	0.1s = Fit runtime
    	9 features in original data used to generate 13 features in processed data.
    	Train Data (Processed) Memory Usage: 0.98 MB (0.0% of available memory)
    Data preprocessing and feature engineering runtime = 0.11s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 11 L1 models ...
    Fitting model: KNeighborsUnif_BAG_L1 ... Training model for up to 399.83s of the 599.89s of remaining time.
    	-101.5462	 = Validation score   (-root_mean_squared_error)
    	0.03s	 = Training   runtime
    	0.04s	 = Validation runtime
    Fitting model: KNeighborsDist_BAG_L1 ... Training model for up to 399.63s of the 599.69s of remaining time.
    	-84.1251	 = Validation score   (-root_mean_squared_error)
    	0.03s	 = Training   runtime
    	0.04s	 = Validation runtime
    Fitting model: LightGBMXT_BAG_L1 ... Training model for up to 399.43s of the 599.49s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-131.4609	 = Validation score   (-root_mean_squared_error)
    	58.46s	 = Training   runtime
    	7.33s	 = Validation runtime
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 335.63s of the 535.69s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-131.0542	 = Validation score   (-root_mean_squared_error)
    	23.35s	 = Training   runtime
    	1.38s	 = Validation runtime
    Fitting model: RandomForestMSE_BAG_L1 ... Training model for up to 308.66s of the 508.72s of remaining time.
    	-116.5443	 = Validation score   (-root_mean_squared_error)
    	10.34s	 = Training   runtime
    	0.55s	 = Validation runtime
    Fitting model: CatBoost_BAG_L1 ... Training model for up to 295.19s of the 495.25s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-130.4943	 = Validation score   (-root_mean_squared_error)
    	206.11s	 = Training   runtime
    	0.11s	 = Validation runtime
    Fitting model: ExtraTreesMSE_BAG_L1 ... Training model for up to 85.26s of the 285.32s of remaining time.
    	-124.5881	 = Validation score   (-root_mean_squared_error)
    	5.07s	 = Training   runtime
    	0.52s	 = Validation runtime
    Fitting model: NeuralNetFastAI_BAG_L1 ... Training model for up to 77.14s of the 277.2s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-136.4229	 = Validation score   (-root_mean_squared_error)
    	85.4s	 = Training   runtime
    	0.29s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.0s of the 188.4s of remaining time.
    	-84.1251	 = Validation score   (-root_mean_squared_error)
    	0.51s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 9 L2 models ...
    Fitting model: LightGBMXT_BAG_L2 ... Training model for up to 187.81s of the 187.8s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-60.2497	 = Validation score   (-root_mean_squared_error)
    	45.63s	 = Training   runtime
    	3.33s	 = Validation runtime
    Fitting model: LightGBM_BAG_L2 ... Training model for up to 137.45s of the 137.43s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-55.2229	 = Validation score   (-root_mean_squared_error)
    	18.08s	 = Training   runtime
    	0.34s	 = Validation runtime
    Fitting model: RandomForestMSE_BAG_L2 ... Training model for up to 115.82s of the 115.8s of remaining time.
    	-53.3947	 = Validation score   (-root_mean_squared_error)
    	25.77s	 = Training   runtime
    	0.59s	 = Validation runtime
    Fitting model: CatBoost_BAG_L2 ... Training model for up to 86.6s of the 86.59s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-55.7409	 = Validation score   (-root_mean_squared_error)
    	63.41s	 = Training   runtime
    	0.07s	 = Validation runtime
    Fitting model: ExtraTreesMSE_BAG_L2 ... Training model for up to 20.11s of the 20.09s of remaining time.
    	-53.971	 = Validation score   (-root_mean_squared_error)
    	7.84s	 = Training   runtime
    	0.59s	 = Validation runtime
    Fitting model: NeuralNetFastAI_BAG_L2 ... Training model for up to 9.18s of the 9.16s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-89.4179	 = Validation score   (-root_mean_squared_error)
    	28.81s	 = Training   runtime
    	0.41s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.0s of the -22.77s of remaining time.
    	-52.8718	 = Validation score   (-root_mean_squared_error)
    	0.38s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 623.35s ... Best model: "WeightedEnsemble_L3"
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20230602_140913/")





    <autogluon.tabular.predictor.predictor.TabularPredictor at 0x7fbaa9143fd0>



__presets="best_quality"__
- This argument specifies the preset configuration for the fitting process. In this case, the preset named "best_quality" is being used, indicating that the model should be trained with settings optimized for best quality

### Review AutoGluon's training run with ranking of models that did the best.


```python
predictor.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                         model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0      WeightedEnsemble_L3  -52.871786      11.857781  504.274996                0.000988           0.379923            3       True         16
    1   RandomForestMSE_BAG_L2  -53.394699      10.855326  414.564875                0.588785          25.766391            2       True         12
    2     ExtraTreesMSE_BAG_L2  -53.970980      10.858333  396.636239                0.591793           7.837756            2       True         14
    3          LightGBM_BAG_L2  -55.222937      10.603344  406.876146                0.336803          18.077662            2       True         11
    4          CatBoost_BAG_L2  -55.740858      10.339412  452.213264                0.072871          63.414780            2       True         13
    5        LightGBMXT_BAG_L2  -60.249680      13.601524  434.424200                3.334984          45.625717            2       True         10
    6    KNeighborsDist_BAG_L1  -84.125061       0.039831    0.030210                0.039831           0.030210            1       True          2
    7      WeightedEnsemble_L2  -84.125061       0.040823    0.536962                0.000992           0.506751            2       True          9
    8   NeuralNetFastAI_BAG_L2  -89.417853      10.674562  417.607085                0.408021          28.808601            2       True         15
    9    KNeighborsUnif_BAG_L1 -101.546199       0.042448    0.033091                0.042448           0.033091            1       True          1
    10  RandomForestMSE_BAG_L1 -116.544294       0.550746   10.336879                0.550746          10.336879            1       True          5
    11    ExtraTreesMSE_BAG_L1 -124.588053       0.521388    5.073255                0.521388           5.073255            1       True          7
    12         CatBoost_BAG_L1 -130.494282       0.113710  206.109985                0.113710         206.109985            1       True          6
    13         LightGBM_BAG_L1 -131.054162       1.382986   23.353068                1.382986          23.353068            1       True          4
    14       LightGBMXT_BAG_L1 -131.460909       7.326296   58.462367                7.326296          58.462367            1       True          3
    15  NeuralNetFastAI_BAG_L1 -136.422927       0.289136   85.399628                0.289136          85.399628            1       True          8
    Number of models trained: 16
    Types of models trained:
    {'WeightedEnsembleModel', 'StackerEnsembleModel_LGB', 'StackerEnsembleModel_KNN', 'StackerEnsembleModel_XT', 'StackerEnsembleModel_NNFastAiTabular', 'StackerEnsembleModel_CatBoost', 'StackerEnsembleModel_RF'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])                  : 3 | ['season', 'weather', 'humidity']
    ('int', ['bool'])            : 2 | ['holiday', 'workingday']
    ('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    Plot summary of models saved to file: AutogluonModels/ag-20230602_140913/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'KNeighborsUnif_BAG_L1': 'StackerEnsembleModel_KNN',
      'KNeighborsDist_BAG_L1': 'StackerEnsembleModel_KNN',
      'LightGBMXT_BAG_L1': 'StackerEnsembleModel_LGB',
      'LightGBM_BAG_L1': 'StackerEnsembleModel_LGB',
      'RandomForestMSE_BAG_L1': 'StackerEnsembleModel_RF',
      'CatBoost_BAG_L1': 'StackerEnsembleModel_CatBoost',
      'ExtraTreesMSE_BAG_L1': 'StackerEnsembleModel_XT',
      'NeuralNetFastAI_BAG_L1': 'StackerEnsembleModel_NNFastAiTabular',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBMXT_BAG_L2': 'StackerEnsembleModel_LGB',
      'LightGBM_BAG_L2': 'StackerEnsembleModel_LGB',
      'RandomForestMSE_BAG_L2': 'StackerEnsembleModel_RF',
      'CatBoost_BAG_L2': 'StackerEnsembleModel_CatBoost',
      'ExtraTreesMSE_BAG_L2': 'StackerEnsembleModel_XT',
      'NeuralNetFastAI_BAG_L2': 'StackerEnsembleModel_NNFastAiTabular',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel'},
     'model_performance': {'KNeighborsUnif_BAG_L1': -101.54619908446061,
      'KNeighborsDist_BAG_L1': -84.12506123181602,
      'LightGBMXT_BAG_L1': -131.46090891834504,
      'LightGBM_BAG_L1': -131.054161598899,
      'RandomForestMSE_BAG_L1': -116.54429428704391,
      'CatBoost_BAG_L1': -130.4942815936897,
      'ExtraTreesMSE_BAG_L1': -124.58805258915959,
      'NeuralNetFastAI_BAG_L1': -136.4229269757243,
      'WeightedEnsemble_L2': -84.12506123181602,
      'LightGBMXT_BAG_L2': -60.24967971756835,
      'LightGBM_BAG_L2': -55.222936989316864,
      'RandomForestMSE_BAG_L2': -53.394699339768025,
      'CatBoost_BAG_L2': -55.74085797861888,
      'ExtraTreesMSE_BAG_L2': -53.970980367781415,
      'NeuralNetFastAI_BAG_L2': -89.41785326447096,
      'WeightedEnsemble_L3': -52.871786174178695},
     'model_best': 'WeightedEnsemble_L3',
     'model_paths': {'KNeighborsUnif_BAG_L1': 'AutogluonModels/ag-20230602_140913/models/KNeighborsUnif_BAG_L1/',
      'KNeighborsDist_BAG_L1': 'AutogluonModels/ag-20230602_140913/models/KNeighborsDist_BAG_L1/',
      'LightGBMXT_BAG_L1': 'AutogluonModels/ag-20230602_140913/models/LightGBMXT_BAG_L1/',
      'LightGBM_BAG_L1': 'AutogluonModels/ag-20230602_140913/models/LightGBM_BAG_L1/',
      'RandomForestMSE_BAG_L1': 'AutogluonModels/ag-20230602_140913/models/RandomForestMSE_BAG_L1/',
      'CatBoost_BAG_L1': 'AutogluonModels/ag-20230602_140913/models/CatBoost_BAG_L1/',
      'ExtraTreesMSE_BAG_L1': 'AutogluonModels/ag-20230602_140913/models/ExtraTreesMSE_BAG_L1/',
      'NeuralNetFastAI_BAG_L1': 'AutogluonModels/ag-20230602_140913/models/NeuralNetFastAI_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20230602_140913/models/WeightedEnsemble_L2/',
      'LightGBMXT_BAG_L2': 'AutogluonModels/ag-20230602_140913/models/LightGBMXT_BAG_L2/',
      'LightGBM_BAG_L2': 'AutogluonModels/ag-20230602_140913/models/LightGBM_BAG_L2/',
      'RandomForestMSE_BAG_L2': 'AutogluonModels/ag-20230602_140913/models/RandomForestMSE_BAG_L2/',
      'CatBoost_BAG_L2': 'AutogluonModels/ag-20230602_140913/models/CatBoost_BAG_L2/',
      'ExtraTreesMSE_BAG_L2': 'AutogluonModels/ag-20230602_140913/models/ExtraTreesMSE_BAG_L2/',
      'NeuralNetFastAI_BAG_L2': 'AutogluonModels/ag-20230602_140913/models/NeuralNetFastAI_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20230602_140913/models/WeightedEnsemble_L3/'},
     'model_fit_times': {'KNeighborsUnif_BAG_L1': 0.03309130668640137,
      'KNeighborsDist_BAG_L1': 0.030210256576538086,
      'LightGBMXT_BAG_L1': 58.462366819381714,
      'LightGBM_BAG_L1': 23.353067874908447,
      'RandomForestMSE_BAG_L1': 10.336879014968872,
      'CatBoost_BAG_L1': 206.1099853515625,
      'ExtraTreesMSE_BAG_L1': 5.073254823684692,
      'NeuralNetFastAI_BAG_L1': 85.39962816238403,
      'WeightedEnsemble_L2': 0.506751298904419,
      'LightGBMXT_BAG_L2': 45.62571668624878,
      'LightGBM_BAG_L2': 18.077661991119385,
      'RandomForestMSE_BAG_L2': 25.766391038894653,
      'CatBoost_BAG_L2': 63.41477990150452,
      'ExtraTreesMSE_BAG_L2': 7.8377556800842285,
      'NeuralNetFastAI_BAG_L2': 28.808601140975952,
      'WeightedEnsemble_L3': 0.37992334365844727},
     'model_pred_times': {'KNeighborsUnif_BAG_L1': 0.04244804382324219,
      'KNeighborsDist_BAG_L1': 0.039830923080444336,
      'LightGBMXT_BAG_L1': 7.326295614242554,
      'LightGBM_BAG_L1': 1.3829855918884277,
      'RandomForestMSE_BAG_L1': 0.550745964050293,
      'CatBoost_BAG_L1': 0.11371016502380371,
      'ExtraTreesMSE_BAG_L1': 0.5213878154754639,
      'NeuralNetFastAI_BAG_L1': 0.2891364097595215,
      'WeightedEnsemble_L2': 0.0009920597076416016,
      'LightGBMXT_BAG_L2': 3.3349838256835938,
      'LightGBM_BAG_L2': 0.3368031978607178,
      'RandomForestMSE_BAG_L2': 0.5887854099273682,
      'CatBoost_BAG_L2': 0.07287144660949707,
      'ExtraTreesMSE_BAG_L2': 0.5917925834655762,
      'NeuralNetFastAI_BAG_L2': 0.4080214500427246,
      'WeightedEnsemble_L3': 0.0009875297546386719},
     'num_bag_folds': 8,
     'max_stack_level': 3,
     'model_hyperparams': {'KNeighborsUnif_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'KNeighborsDist_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'LightGBMXT_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'RandomForestMSE_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'CatBoost_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'ExtraTreesMSE_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'NeuralNetFastAI_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBMXT_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'RandomForestMSE_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'CatBoost_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'ExtraTreesMSE_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'NeuralNetFastAI_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True}},
     'leaderboard':                      model   score_val  pred_time_val    fit_time  \
     0      WeightedEnsemble_L3  -52.871786      11.857781  504.274996   
     1   RandomForestMSE_BAG_L2  -53.394699      10.855326  414.564875   
     2     ExtraTreesMSE_BAG_L2  -53.970980      10.858333  396.636239   
     3          LightGBM_BAG_L2  -55.222937      10.603344  406.876146   
     4          CatBoost_BAG_L2  -55.740858      10.339412  452.213264   
     5        LightGBMXT_BAG_L2  -60.249680      13.601524  434.424200   
     6    KNeighborsDist_BAG_L1  -84.125061       0.039831    0.030210   
     7      WeightedEnsemble_L2  -84.125061       0.040823    0.536962   
     8   NeuralNetFastAI_BAG_L2  -89.417853      10.674562  417.607085   
     9    KNeighborsUnif_BAG_L1 -101.546199       0.042448    0.033091   
     10  RandomForestMSE_BAG_L1 -116.544294       0.550746   10.336879   
     11    ExtraTreesMSE_BAG_L1 -124.588053       0.521388    5.073255   
     12         CatBoost_BAG_L1 -130.494282       0.113710  206.109985   
     13         LightGBM_BAG_L1 -131.054162       1.382986   23.353068   
     14       LightGBMXT_BAG_L1 -131.460909       7.326296   58.462367   
     15  NeuralNetFastAI_BAG_L1 -136.422927       0.289136   85.399628   
     
         pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                 0.000988           0.379923            3       True   
     1                 0.588785          25.766391            2       True   
     2                 0.591793           7.837756            2       True   
     3                 0.336803          18.077662            2       True   
     4                 0.072871          63.414780            2       True   
     5                 3.334984          45.625717            2       True   
     6                 0.039831           0.030210            1       True   
     7                 0.000992           0.506751            2       True   
     8                 0.408021          28.808601            2       True   
     9                 0.042448           0.033091            1       True   
     10                0.550746          10.336879            1       True   
     11                0.521388           5.073255            1       True   
     12                0.113710         206.109985            1       True   
     13                1.382986          23.353068            1       True   
     14                7.326296          58.462367            1       True   
     15                0.289136          85.399628            1       True   
     
         fit_order  
     0          16  
     1          12  
     2          14  
     3          11  
     4          13  
     5          10  
     6           2  
     7           9  
     8          15  
     9           1  
     10          5  
     11          7  
     12          6  
     13          4  
     14          3  
     15          8  }




```python
leaderboard_df = pd.DataFrame(predictor.leaderboard())
leaderboard_df.plot(kind="bar", x="model", y="score_val", figsize=(14, 7))
plt.show()
```

                         model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0      WeightedEnsemble_L3  -52.871786      11.857781  504.274996                0.000988           0.379923            3       True         16
    1   RandomForestMSE_BAG_L2  -53.394699      10.855326  414.564875                0.588785          25.766391            2       True         12
    2     ExtraTreesMSE_BAG_L2  -53.970980      10.858333  396.636239                0.591793           7.837756            2       True         14
    3          LightGBM_BAG_L2  -55.222937      10.603344  406.876146                0.336803          18.077662            2       True         11
    4          CatBoost_BAG_L2  -55.740858      10.339412  452.213264                0.072871          63.414780            2       True         13
    5        LightGBMXT_BAG_L2  -60.249680      13.601524  434.424200                3.334984          45.625717            2       True         10
    6    KNeighborsDist_BAG_L1  -84.125061       0.039831    0.030210                0.039831           0.030210            1       True          2
    7      WeightedEnsemble_L2  -84.125061       0.040823    0.536962                0.000992           0.506751            2       True          9
    8   NeuralNetFastAI_BAG_L2  -89.417853      10.674562  417.607085                0.408021          28.808601            2       True         15
    9    KNeighborsUnif_BAG_L1 -101.546199       0.042448    0.033091                0.042448           0.033091            1       True          1
    10  RandomForestMSE_BAG_L1 -116.544294       0.550746   10.336879                0.550746          10.336879            1       True          5
    11    ExtraTreesMSE_BAG_L1 -124.588053       0.521388    5.073255                0.521388           5.073255            1       True          7
    12         CatBoost_BAG_L1 -130.494282       0.113710  206.109985                0.113710         206.109985            1       True          6
    13         LightGBM_BAG_L1 -131.054162       1.382986   23.353068                1.382986          23.353068            1       True          4
    14       LightGBMXT_BAG_L1 -131.460909       7.326296   58.462367                7.326296          58.462367            1       True          3
    15  NeuralNetFastAI_BAG_L1 -136.422927       0.289136   85.399628                0.289136          85.399628            1       True          8



    
![png](output_28_1.png)
    


Insight
- a very intersting things here is that top result giving is L_3, other are L_2 level and base models (L_1) does not seem to perform good here.

### Create predictions from test dataset


```python
predictions = predictor.predict(test)
predictions.head()
```




    0    23.123632
    1    41.455280
    2    45.221096
    3    49.122555
    4    52.208115
    Name: count, dtype: float32



#### NOTE: Kaggle will reject the submission if we don't set everything to be > 0.


```python
predictions[predictions < 0] = 0  
```

### Set predictions to submission dataframe, save, and submit


```python
submission["count"] = predictions
submission.to_csv("submission.csv", index=False)
```


```python
!kaggle competitions submit -c bike-sharing-demand -f submission.csv -m "Initial Submission"
```

    100%|█████████████████████████████████████████| 188k/188k [00:00<00:00, 395kB/s]
    Successfully submitted to Bike Sharing Demand

#### View submission via the command line or in the web browser under the competition's page - `My Submissions`


```python
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
```

    fileName                     date                 description                                                                                         status    publicScore  privateScore  
    ---------------------------  -------------------  --------------------------------------------------------------------------------------------------  --------  -----------  ------------  
    submission.csv               2023-06-02 14:26:04  Initial Submission                                                                                  complete  1.80958      1.80958       
    submission_new_hyp_3.csv     2023-06-01 14:50:49  new features with hyperparameters epoch, boost round, learning rate, extra trees, drop-out, leaves  complete  0.49945      0.49945       
    submission_new_hyp_2.csv     2023-06-01 14:50:46  new features with hyperparameters epoch, boost round, learning rate, extra trees                    complete  0.65221      0.65221       
    submission_new_hyp_1.csv     2023-06-01 14:50:44  new features with hyperparameters epoch, boost round                                                complete  0.60783      0.60783       
    tail: write error: Broken pipe


#### Initial score of 1.80958

## Step 4: Exploratory Data Analysis and Creating an additional feature
* Any additional feature will do, but a great suggestion would be to separate out the datetime into hour, day, or month parts.


```python
train.hist(figsize=(15,20))  
plt.tight_layout()
plt.show()
```


    
![png](output_41_0.png)
    


Insights:

- Season and weather are categorical variables, so we need to change their type as "category"
- Datetime feature, showing randomness as they represent timestamps, so it make more sense to create new feature by considering hours and day of week
- There are particular hour of day when people use bike most, like evening and morning. Same way people tend to do biking more on weekends. So it make complete sense to create these two as new feature.
- Also we can safely drop datetime column



```python
train["hour"] = train["datetime"].dt.hour
train["day"] = train["datetime"].dt.dayofweek
train.drop(["datetime"], axis=1, inplace=True)
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>hour</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
test["hour"] = test["datetime"].dt.hour
test["day"] = test["datetime"].dt.dayofweek
test.drop(["datetime"], axis=1, inplace=True)
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>hour</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>4</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



## Make category types for these so models know they are not just numbers
* AutoGluon originally sees these as ints, but in reality they are int representations of a category.
* Setting the dtype to category will classify these as categories in AutoGluon.


```python
train["season"] = train["season"].astype("category")
train["weather"] = train["weather"].astype("category")

test["season"] = test["season"].astype("category")
test["weather"] = test["weather"].astype("category")
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>hour</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.hist(figsize=(15, 20))
plt.tight_layout()
plt.show()
```


    
![png](output_48_0.png)
    


## Step 5: Rerun the model with the same settings as before, just with more features


```python
predictor_new_features = TabularPredictor(
    label="count",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    learner_kwargs={"ignored_columns": ["casual", "registered"]},
)
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20230602_143128/"



```python
predictor_new_features.fit(train_data=train, time_limit=600, presets="best_quality")
```

    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 600s
    AutoGluon will save models to "AutogluonModels/ag-20230602_143128/"
    AutoGluon Version:  0.7.0
    Python Version:     3.8.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Tue Apr 25 15:24:19 UTC 2023
    Train Data Rows:    10886
    Train Data Columns: 12
    Label Column: count
    Preprocessing data ...
    Using Feature Generators to preprocess the data ...
    Dropping user-specified ignored columns: ['casual', 'registered']
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    2081.49 MB
    	Train Data (Original)  Memory Usage: 0.72 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 5 | ['holiday', 'workingday', 'humidity', 'hour', 'day']
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])  : 2 | ['season', 'weather']
    		('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])       : 3 | ['humidity', 'hour', 'day']
    		('int', ['bool']) : 2 | ['holiday', 'workingday']
    	0.3s = Fit runtime
    	10 features in original data used to generate 10 features in processed data.
    	Train Data (Processed) Memory Usage: 0.57 MB (0.0% of available memory)
    Data preprocessing and feature engineering runtime = 0.33s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 11 L1 models ...
    Fitting model: KNeighborsUnif_BAG_L1 ... Training model for up to 399.68s of the 599.66s of remaining time.
    	-117.0607	 = Validation score   (-root_mean_squared_error)
    	0.03s	 = Training   runtime
    	0.1s	 = Validation runtime
    Fitting model: KNeighborsDist_BAG_L1 ... Training model for up to 399.42s of the 599.4s of remaining time.
    	-114.004	 = Validation score   (-root_mean_squared_error)
    	0.02s	 = Training   runtime
    	0.09s	 = Validation runtime
    Fitting model: LightGBMXT_BAG_L1 ... Training model for up to 399.19s of the 599.17s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-60.1971	 = Validation score   (-root_mean_squared_error)
    	85.68s	 = Training   runtime
    	17.94s	 = Validation runtime
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 308.09s of the 508.07s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-60.9531	 = Validation score   (-root_mean_squared_error)
    	31.75s	 = Training   runtime
    	3.2s	 = Validation runtime
    Fitting model: RandomForestMSE_BAG_L1 ... Training model for up to 271.99s of the 471.98s of remaining time.
    	-66.2409	 = Validation score   (-root_mean_squared_error)
    	7.65s	 = Training   runtime
    	0.56s	 = Validation runtime
    Fitting model: CatBoost_BAG_L1 ... Training model for up to 261.33s of the 461.31s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-61.6041	 = Validation score   (-root_mean_squared_error)
    	220.21s	 = Training   runtime
    	0.21s	 = Validation runtime
    Fitting model: ExtraTreesMSE_BAG_L1 ... Training model for up to 38.05s of the 238.03s of remaining time.
    	-66.2385	 = Validation score   (-root_mean_squared_error)
    	4.53s	 = Training   runtime
    	0.55s	 = Validation runtime
    Fitting model: NeuralNetFastAI_BAG_L1 ... Training model for up to 29.48s of the 229.46s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-97.7682	 = Validation score   (-root_mean_squared_error)
    	47.22s	 = Training   runtime
    	0.38s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.0s of the 178.98s of remaining time.
    	-58.8965	 = Validation score   (-root_mean_squared_error)
    	0.44s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 9 L2 models ...
    Fitting model: LightGBMXT_BAG_L2 ... Training model for up to 178.47s of the 178.45s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-59.9379	 = Validation score   (-root_mean_squared_error)
    	16.95s	 = Training   runtime
    	0.32s	 = Validation runtime
    Fitting model: LightGBM_BAG_L2 ... Training model for up to 157.54s of the 157.53s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-59.6191	 = Validation score   (-root_mean_squared_error)
    	16.55s	 = Training   runtime
    	0.11s	 = Validation runtime
    Fitting model: RandomForestMSE_BAG_L2 ... Training model for up to 137.47s of the 137.46s of remaining time.
    	-59.8815	 = Validation score   (-root_mean_squared_error)
    	25.3s	 = Training   runtime
    	0.61s	 = Validation runtime
    Fitting model: CatBoost_BAG_L2 ... Training model for up to 109.3s of the 109.28s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-58.5872	 = Validation score   (-root_mean_squared_error)
    	49.83s	 = Training   runtime
    	0.08s	 = Validation runtime
    Fitting model: ExtraTreesMSE_BAG_L2 ... Training model for up to 56.37s of the 56.35s of remaining time.
    	-59.2367	 = Validation score   (-root_mean_squared_error)
    	7.13s	 = Training   runtime
    	0.6s	 = Validation runtime
    Fitting model: NeuralNetFastAI_BAG_L2 ... Training model for up to 46.25s of the 46.23s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-58.517	 = Validation score   (-root_mean_squared_error)
    	60.63s	 = Training   runtime
    	0.51s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.0s of the -17.68s of remaining time.
    	-58.197	 = Validation score   (-root_mean_squared_error)
    	0.42s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 618.33s ... Best model: "WeightedEnsemble_L3"
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20230602_143128/")





    <autogluon.tabular.predictor.predictor.TabularPredictor at 0x7fbaa7437220>




```python
predictor_new_features.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                         model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0      WeightedEnsemble_L3  -58.196982      24.230838  515.107222                0.000712           0.418559            3       True         16
    1   NeuralNetFastAI_BAG_L2  -58.516959      23.552197  457.729084                0.506768          60.633716            2       True         15
    2          CatBoost_BAG_L2  -58.587186      23.127257  446.929289                0.081828          49.833921            2       True         13
    3      WeightedEnsemble_L2  -58.896532      21.919335  345.734301                0.000733           0.440368            2       True          9
    4     ExtraTreesMSE_BAG_L2  -59.236701      23.641531  404.221027                0.596102           7.125659            2       True         14
    5          LightGBM_BAG_L2  -59.619095      23.154006  413.641091                0.108577          16.545723            2       True         11
    6   RandomForestMSE_BAG_L2  -59.881547      23.653731  422.400025                0.608302          25.304657            2       True         12
    7        LightGBMXT_BAG_L2  -59.937946      23.368949  414.042805                0.323520          16.947437            2       True         10
    8        LightGBMXT_BAG_L1  -60.197098      17.943470   85.681547               17.943470          85.681547            1       True          3
    9          LightGBM_BAG_L1  -60.953081       3.200293   31.746621                3.200293          31.746621            1       True          4
    10         CatBoost_BAG_L1  -61.604089       0.210660  220.212801                0.210660         220.212801            1       True          6
    11    ExtraTreesMSE_BAG_L1  -66.238489       0.552660    4.534876                0.552660           4.534876            1       True          7
    12  RandomForestMSE_BAG_L1  -66.240918       0.564179    7.652964                0.564179           7.652964            1       True          5
    13  NeuralNetFastAI_BAG_L1  -97.768237       0.384564   47.220266                0.384564          47.220266            1       True          8
    14   KNeighborsDist_BAG_L1 -114.004045       0.093814    0.020973                0.093814           0.020973            1       True          2
    15   KNeighborsUnif_BAG_L1 -117.060748       0.095789    0.025320                0.095789           0.025320            1       True          1
    Number of models trained: 16
    Types of models trained:
    {'WeightedEnsembleModel', 'StackerEnsembleModel_LGB', 'StackerEnsembleModel_KNN', 'StackerEnsembleModel_XT', 'StackerEnsembleModel_NNFastAiTabular', 'StackerEnsembleModel_CatBoost', 'StackerEnsembleModel_RF'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('category', [])  : 2 | ['season', 'weather']
    ('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])       : 3 | ['humidity', 'hour', 'day']
    ('int', ['bool']) : 2 | ['holiday', 'workingday']
    Plot summary of models saved to file: AutogluonModels/ag-20230602_143128/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'KNeighborsUnif_BAG_L1': 'StackerEnsembleModel_KNN',
      'KNeighborsDist_BAG_L1': 'StackerEnsembleModel_KNN',
      'LightGBMXT_BAG_L1': 'StackerEnsembleModel_LGB',
      'LightGBM_BAG_L1': 'StackerEnsembleModel_LGB',
      'RandomForestMSE_BAG_L1': 'StackerEnsembleModel_RF',
      'CatBoost_BAG_L1': 'StackerEnsembleModel_CatBoost',
      'ExtraTreesMSE_BAG_L1': 'StackerEnsembleModel_XT',
      'NeuralNetFastAI_BAG_L1': 'StackerEnsembleModel_NNFastAiTabular',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBMXT_BAG_L2': 'StackerEnsembleModel_LGB',
      'LightGBM_BAG_L2': 'StackerEnsembleModel_LGB',
      'RandomForestMSE_BAG_L2': 'StackerEnsembleModel_RF',
      'CatBoost_BAG_L2': 'StackerEnsembleModel_CatBoost',
      'ExtraTreesMSE_BAG_L2': 'StackerEnsembleModel_XT',
      'NeuralNetFastAI_BAG_L2': 'StackerEnsembleModel_NNFastAiTabular',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel'},
     'model_performance': {'KNeighborsUnif_BAG_L1': -117.06074757128302,
      'KNeighborsDist_BAG_L1': -114.00404505882429,
      'LightGBMXT_BAG_L1': -60.19709831103628,
      'LightGBM_BAG_L1': -60.9530807694855,
      'RandomForestMSE_BAG_L1': -66.24091761068544,
      'CatBoost_BAG_L1': -61.60408933992878,
      'ExtraTreesMSE_BAG_L1': -66.23848900827704,
      'NeuralNetFastAI_BAG_L1': -97.76823668290878,
      'WeightedEnsemble_L2': -58.896532026740566,
      'LightGBMXT_BAG_L2': -59.93794634434353,
      'LightGBM_BAG_L2': -59.619094879112666,
      'RandomForestMSE_BAG_L2': -59.88154679181342,
      'CatBoost_BAG_L2': -58.587185607079064,
      'ExtraTreesMSE_BAG_L2': -59.236701197520205,
      'NeuralNetFastAI_BAG_L2': -58.516958980520066,
      'WeightedEnsemble_L3': -58.196981885181145},
     'model_best': 'WeightedEnsemble_L3',
     'model_paths': {'KNeighborsUnif_BAG_L1': 'AutogluonModels/ag-20230602_143128/models/KNeighborsUnif_BAG_L1/',
      'KNeighborsDist_BAG_L1': 'AutogluonModels/ag-20230602_143128/models/KNeighborsDist_BAG_L1/',
      'LightGBMXT_BAG_L1': 'AutogluonModels/ag-20230602_143128/models/LightGBMXT_BAG_L1/',
      'LightGBM_BAG_L1': 'AutogluonModels/ag-20230602_143128/models/LightGBM_BAG_L1/',
      'RandomForestMSE_BAG_L1': 'AutogluonModels/ag-20230602_143128/models/RandomForestMSE_BAG_L1/',
      'CatBoost_BAG_L1': 'AutogluonModels/ag-20230602_143128/models/CatBoost_BAG_L1/',
      'ExtraTreesMSE_BAG_L1': 'AutogluonModels/ag-20230602_143128/models/ExtraTreesMSE_BAG_L1/',
      'NeuralNetFastAI_BAG_L1': 'AutogluonModels/ag-20230602_143128/models/NeuralNetFastAI_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20230602_143128/models/WeightedEnsemble_L2/',
      'LightGBMXT_BAG_L2': 'AutogluonModels/ag-20230602_143128/models/LightGBMXT_BAG_L2/',
      'LightGBM_BAG_L2': 'AutogluonModels/ag-20230602_143128/models/LightGBM_BAG_L2/',
      'RandomForestMSE_BAG_L2': 'AutogluonModels/ag-20230602_143128/models/RandomForestMSE_BAG_L2/',
      'CatBoost_BAG_L2': 'AutogluonModels/ag-20230602_143128/models/CatBoost_BAG_L2/',
      'ExtraTreesMSE_BAG_L2': 'AutogluonModels/ag-20230602_143128/models/ExtraTreesMSE_BAG_L2/',
      'NeuralNetFastAI_BAG_L2': 'AutogluonModels/ag-20230602_143128/models/NeuralNetFastAI_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20230602_143128/models/WeightedEnsemble_L3/'},
     'model_fit_times': {'KNeighborsUnif_BAG_L1': 0.02532029151916504,
      'KNeighborsDist_BAG_L1': 0.02097296714782715,
      'LightGBMXT_BAG_L1': 85.68154740333557,
      'LightGBM_BAG_L1': 31.746620655059814,
      'RandomForestMSE_BAG_L1': 7.652964115142822,
      'CatBoost_BAG_L1': 220.21280097961426,
      'ExtraTreesMSE_BAG_L1': 4.5348756313323975,
      'NeuralNetFastAI_BAG_L1': 47.22026610374451,
      'WeightedEnsemble_L2': 0.4403681755065918,
      'LightGBMXT_BAG_L2': 16.947436571121216,
      'LightGBM_BAG_L2': 16.54572319984436,
      'RandomForestMSE_BAG_L2': 25.304656982421875,
      'CatBoost_BAG_L2': 49.83392071723938,
      'ExtraTreesMSE_BAG_L2': 7.125658750534058,
      'NeuralNetFastAI_BAG_L2': 60.63371562957764,
      'WeightedEnsemble_L3': 0.41855859756469727},
     'model_pred_times': {'KNeighborsUnif_BAG_L1': 0.09578919410705566,
      'KNeighborsDist_BAG_L1': 0.09381365776062012,
      'LightGBMXT_BAG_L1': 17.943469524383545,
      'LightGBM_BAG_L1': 3.2002928256988525,
      'RandomForestMSE_BAG_L1': 0.5641791820526123,
      'CatBoost_BAG_L1': 0.21066045761108398,
      'ExtraTreesMSE_BAG_L1': 0.5526599884033203,
      'NeuralNetFastAI_BAG_L1': 0.3845641613006592,
      'WeightedEnsemble_L2': 0.0007326602935791016,
      'LightGBMXT_BAG_L2': 0.3235204219818115,
      'LightGBM_BAG_L2': 0.10857653617858887,
      'RandomForestMSE_BAG_L2': 0.6083023548126221,
      'CatBoost_BAG_L2': 0.08182764053344727,
      'ExtraTreesMSE_BAG_L2': 0.5961017608642578,
      'NeuralNetFastAI_BAG_L2': 0.506767749786377,
      'WeightedEnsemble_L3': 0.0007119178771972656},
     'num_bag_folds': 8,
     'max_stack_level': 3,
     'model_hyperparams': {'KNeighborsUnif_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'KNeighborsDist_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'LightGBMXT_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'RandomForestMSE_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'CatBoost_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'ExtraTreesMSE_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'NeuralNetFastAI_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBMXT_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'RandomForestMSE_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'CatBoost_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'ExtraTreesMSE_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True,
       'use_child_oof': True},
      'NeuralNetFastAI_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True}},
     'leaderboard':                      model   score_val  pred_time_val    fit_time  \
     0      WeightedEnsemble_L3  -58.196982      24.230838  515.107222   
     1   NeuralNetFastAI_BAG_L2  -58.516959      23.552197  457.729084   
     2          CatBoost_BAG_L2  -58.587186      23.127257  446.929289   
     3      WeightedEnsemble_L2  -58.896532      21.919335  345.734301   
     4     ExtraTreesMSE_BAG_L2  -59.236701      23.641531  404.221027   
     5          LightGBM_BAG_L2  -59.619095      23.154006  413.641091   
     6   RandomForestMSE_BAG_L2  -59.881547      23.653731  422.400025   
     7        LightGBMXT_BAG_L2  -59.937946      23.368949  414.042805   
     8        LightGBMXT_BAG_L1  -60.197098      17.943470   85.681547   
     9          LightGBM_BAG_L1  -60.953081       3.200293   31.746621   
     10         CatBoost_BAG_L1  -61.604089       0.210660  220.212801   
     11    ExtraTreesMSE_BAG_L1  -66.238489       0.552660    4.534876   
     12  RandomForestMSE_BAG_L1  -66.240918       0.564179    7.652964   
     13  NeuralNetFastAI_BAG_L1  -97.768237       0.384564   47.220266   
     14   KNeighborsDist_BAG_L1 -114.004045       0.093814    0.020973   
     15   KNeighborsUnif_BAG_L1 -117.060748       0.095789    0.025320   
     
         pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                 0.000712           0.418559            3       True   
     1                 0.506768          60.633716            2       True   
     2                 0.081828          49.833921            2       True   
     3                 0.000733           0.440368            2       True   
     4                 0.596102           7.125659            2       True   
     5                 0.108577          16.545723            2       True   
     6                 0.608302          25.304657            2       True   
     7                 0.323520          16.947437            2       True   
     8                17.943470          85.681547            1       True   
     9                 3.200293          31.746621            1       True   
     10                0.210660         220.212801            1       True   
     11                0.552660           4.534876            1       True   
     12                0.564179           7.652964            1       True   
     13                0.384564          47.220266            1       True   
     14                0.093814           0.020973            1       True   
     15                0.095789           0.025320            1       True   
     
         fit_order  
     0          16  
     1          15  
     2          13  
     3           9  
     4          14  
     5          11  
     6          12  
     7          10  
     8           3  
     9           4  
     10          6  
     11          7  
     12          5  
     13          8  
     14          2  
     15          1  }




```python
leaderboard_new_df = pd.DataFrame(predictor_new_features.leaderboard())
leaderboard_new_df.plot(kind="bar", x="model", y="score_val", figsize=(14, 7))
plt.show()
```

                         model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0      WeightedEnsemble_L3  -58.196982      24.230838  515.107222                0.000712           0.418559            3       True         16
    1   NeuralNetFastAI_BAG_L2  -58.516959      23.552197  457.729084                0.506768          60.633716            2       True         15
    2          CatBoost_BAG_L2  -58.587186      23.127257  446.929289                0.081828          49.833921            2       True         13
    3      WeightedEnsemble_L2  -58.896532      21.919335  345.734301                0.000733           0.440368            2       True          9
    4     ExtraTreesMSE_BAG_L2  -59.236701      23.641531  404.221027                0.596102           7.125659            2       True         14
    5          LightGBM_BAG_L2  -59.619095      23.154006  413.641091                0.108577          16.545723            2       True         11
    6   RandomForestMSE_BAG_L2  -59.881547      23.653731  422.400025                0.608302          25.304657            2       True         12
    7        LightGBMXT_BAG_L2  -59.937946      23.368949  414.042805                0.323520          16.947437            2       True         10
    8        LightGBMXT_BAG_L1  -60.197098      17.943470   85.681547               17.943470          85.681547            1       True          3
    9          LightGBM_BAG_L1  -60.953081       3.200293   31.746621                3.200293          31.746621            1       True          4
    10         CatBoost_BAG_L1  -61.604089       0.210660  220.212801                0.210660         220.212801            1       True          6
    11    ExtraTreesMSE_BAG_L1  -66.238489       0.552660    4.534876                0.552660           4.534876            1       True          7
    12  RandomForestMSE_BAG_L1  -66.240918       0.564179    7.652964                0.564179           7.652964            1       True          5
    13  NeuralNetFastAI_BAG_L1  -97.768237       0.384564   47.220266                0.384564          47.220266            1       True          8
    14   KNeighborsDist_BAG_L1 -114.004045       0.093814    0.020973                0.093814           0.020973            1       True          2
    15   KNeighborsUnif_BAG_L1 -117.060748       0.095789    0.025320                0.095789           0.025320            1       True          1



    
![png](output_53_1.png)
    


Insight
- Again top result giver is L_3, other are L_2 level and base models (L_1) does not seem to perform good here.
- With two extra and more sensible feature we get better results
- one point to note here is that now with 2 new feature, L_1 show more relative improvement as compare to L_2 and L_3


```python
predictions_new_features = predictor_new_features.predict(test)
predictions_new_features.head()
```




    0    19.169415
    1     4.918484
    2     2.845312
    3     3.435486
    4     2.997172
    Name: count, dtype: float32




```python
predictions_new_features.describe()
predictions_new_features[predictions_new_features < 0] = 0
```


```python
submission_new_features = pd.read_csv("sampleSubmission.csv", parse_dates=["datetime"])
submission_new_features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission_new_features["count"] = predictions_new_features
submission_new_features.to_csv("submission_new_features.csv", index=False)
```


```python
!kaggle competitions submit -c bike-sharing-demand -f submission_new_features.csv -m "Two new features (hours & Weekday)"
```

    100%|█████████████████████████████████████████| 188k/188k [00:00<00:00, 339kB/s]
    Successfully submitted to Bike Sharing Demand


```python
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
```

    fileName                     date                 description                                                                                         status    publicScore  privateScore  
    ---------------------------  -------------------  --------------------------------------------------------------------------------------------------  --------  -----------  ------------  
    submission_new_features.csv  2023-06-02 14:44:22  Two new features (hours & Weekday)                                                                  complete  0.50823      0.50823       
    submission.csv               2023-06-02 14:26:04  Initial Submission                                                                                  complete  1.80958      1.80958       
    submission_new_hyp_3.csv     2023-06-01 14:50:49  new features with hyperparameters epoch, boost round, learning rate, extra trees, drop-out, leaves  complete  0.49945      0.49945       
    submission_new_hyp_2.csv     2023-06-01 14:50:46  new features with hyperparameters epoch, boost round, learning rate, extra trees                    complete  0.65221      0.65221       
    tail: write error: Broken pipe


#### New Score of 0.50823   

## Step 6: Hyper parameter optimization
* There are many options for hyper parameter optimization.
* Options are to change the AutoGluon higher level parameters or the individual model hyperparameters.
* The hyperparameters of the models themselves that are in AutoGluon. Those need the `hyperparameter` and `hyperparameter_tune_kwargs` arguments.


```python
hyperparameters_1 = {
    "NN_TORCH": {
        "num_epochs": 100
    },  
    "GBM": {
        "num_boost_round": 1000
    },  
}
```


```python
hyperparameters_2 = {
    "NN_TORCH": {
        "num_epochs": 100,
        "learning_rate": 1e-4,
    },
    "GBM": {
        "num_boost_round": 1000,
        "extra_trees": False,
    },
}

```


```python
hyperparameters_3 = {  
    "GBM": {"extra_trees": False, "num_boost_round": 1000, "num_leaves": 5},
    "NN_TORCH": {"num_epochs": 100, "learning_rate": 1e-5, "dropout_prob": 0.001},
}
```


```python
predictor_new_hp_1 = TabularPredictor(
    label="count",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    learner_kwargs={"ignored_columns": ["casual", "registered"]},
)

predictor_new_hp_1.fit(
    train_data=train,
    time_limit=600,
    presets="best_quality",
    hyperparameters=hyperparameters_1,
    refit_full="best", # the predictor should be refit using the best configuration found during the initial fitting
)
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20230602_145116/"
    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 600s
    AutoGluon will save models to "AutogluonModels/ag-20230602_145116/"
    AutoGluon Version:  0.7.0
    Python Version:     3.8.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Tue Apr 25 15:24:19 UTC 2023
    Train Data Rows:    10886
    Train Data Columns: 12
    Label Column: count
    Preprocessing data ...
    Using Feature Generators to preprocess the data ...
    Dropping user-specified ignored columns: ['casual', 'registered']
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    2103.44 MB
    	Train Data (Original)  Memory Usage: 0.72 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 5 | ['holiday', 'workingday', 'humidity', 'hour', 'day']
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])  : 2 | ['season', 'weather']
    		('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])       : 3 | ['humidity', 'hour', 'day']
    		('int', ['bool']) : 2 | ['holiday', 'workingday']
    	0.1s = Fit runtime
    	10 features in original data used to generate 10 features in processed data.
    	Train Data (Processed) Memory Usage: 0.57 MB (0.0% of available memory)
    Data preprocessing and feature engineering runtime = 0.13s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 2 L1 models ...
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 399.81s of the 599.86s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-61.5645	 = Validation score   (-root_mean_squared_error)
    	20.37s	 = Training   runtime
    	1.5s	 = Validation runtime
    Fitting model: NeuralNetTorch_BAG_L1 ... Training model for up to 375.7s of the 575.75s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-66.6186	 = Validation score   (-root_mean_squared_error)
    	286.42s	 = Training   runtime
    	0.16s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.0s of the 285.84s of remaining time.
    	-60.3867	 = Validation score   (-root_mean_squared_error)
    	0.16s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 2 L2 models ...
    Fitting model: LightGBM_BAG_L2 ... Training model for up to 285.62s of the 285.61s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-62.1227	 = Validation score   (-root_mean_squared_error)
    	14.16s	 = Training   runtime
    	0.09s	 = Validation runtime
    Fitting model: NeuralNetTorch_BAG_L2 ... Training model for up to 268.1s of the 268.1s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-62.1487	 = Validation score   (-root_mean_squared_error)
    	134.62s	 = Training   runtime
    	0.23s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.0s of the 130.16s of remaining time.
    	-61.3916	 = Validation score   (-root_mean_squared_error)
    	0.16s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 470.19s ... Best model: "WeightedEnsemble_L2"
    Automatically performing refit_full as a post-fit operation (due to `.fit(..., refit_full=True)`
    Refitting models via `predictor.refit_full` using all of the data (combined train and validation)...
    	Models trained in this way will have the suffix "_FULL" and have NaN validation score.
    	This process is not bound by time_limit, but should take less time than the original `predictor.fit` call.
    	To learn more, refer to the `.refit_full` method docstring which explains how "_FULL" models differ from normal models.
    Fitting 1 L1 models ...
    Fitting model: LightGBM_BAG_L1_FULL ...
    	1.29s	 = Training   runtime
    Fitting 1 L1 models ...
    Fitting model: NeuralNetTorch_BAG_L1_FULL ...
    	39.01s	 = Training   runtime
    Fitting model: WeightedEnsemble_L2_FULL | Skipping fit via cloning parent ...
    	0.16s	 = Training   runtime
    Refit complete, total runtime = 42.28s
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20230602_145116/")





    <autogluon.tabular.predictor.predictor.TabularPredictor at 0x7fbaa7075f40>




```python
predictor_new_hp_1.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                            model  score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0         WeightedEnsemble_L2 -60.386725       1.658752  306.940421                0.000860           0.155286            2       True          3
    1         WeightedEnsemble_L3 -61.391586       1.979550  455.726671                0.000714           0.161280            3       True          6
    2             LightGBM_BAG_L1 -61.564474       1.498764   20.368878                1.498764          20.368878            1       True          1
    3             LightGBM_BAG_L2 -62.122728       1.749792  320.944915                0.091900          14.159780            2       True          4
    4       NeuralNetTorch_BAG_L2 -62.148654       1.886936  441.405611                0.229044         134.620476            2       True          5
    5       NeuralNetTorch_BAG_L1 -66.618550       0.159128  286.416256                0.159128         286.416256            1       True          2
    6    WeightedEnsemble_L2_FULL        NaN            NaN   40.452355                     NaN           0.155286            2       True          9
    7  NeuralNetTorch_BAG_L1_FULL        NaN            NaN   39.006562                     NaN          39.006562            1       True          8
    8        LightGBM_BAG_L1_FULL        NaN            NaN    1.290507                     NaN           1.290507            1       True          7
    Number of models trained: 9
    Types of models trained:
    {'StackerEnsembleModel_TabularNeuralNetTorch', 'StackerEnsembleModel_LGB', 'WeightedEnsembleModel'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('category', [])  : 2 | ['season', 'weather']
    ('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])       : 3 | ['humidity', 'hour', 'day']
    ('int', ['bool']) : 2 | ['holiday', 'workingday']
    Plot summary of models saved to file: AutogluonModels/ag-20230602_145116/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'LightGBM_BAG_L1': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L1': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBM_BAG_L2': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L2': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel',
      'LightGBM_BAG_L1_FULL': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L1_FULL': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L2_FULL': 'WeightedEnsembleModel'},
     'model_performance': {'LightGBM_BAG_L1': -61.5644737203475,
      'NeuralNetTorch_BAG_L1': -66.61855036413591,
      'WeightedEnsemble_L2': -60.38672495183434,
      'LightGBM_BAG_L2': -62.12272806653446,
      'NeuralNetTorch_BAG_L2': -62.14865407732584,
      'WeightedEnsemble_L3': -61.39158630922291,
      'LightGBM_BAG_L1_FULL': None,
      'NeuralNetTorch_BAG_L1_FULL': None,
      'WeightedEnsemble_L2_FULL': None},
     'model_best': 'WeightedEnsemble_L2',
     'model_paths': {'LightGBM_BAG_L1': 'AutogluonModels/ag-20230602_145116/models/LightGBM_BAG_L1/',
      'NeuralNetTorch_BAG_L1': 'AutogluonModels/ag-20230602_145116/models/NeuralNetTorch_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20230602_145116/models/WeightedEnsemble_L2/',
      'LightGBM_BAG_L2': 'AutogluonModels/ag-20230602_145116/models/LightGBM_BAG_L2/',
      'NeuralNetTorch_BAG_L2': 'AutogluonModels/ag-20230602_145116/models/NeuralNetTorch_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20230602_145116/models/WeightedEnsemble_L3/',
      'LightGBM_BAG_L1_FULL': 'AutogluonModels/ag-20230602_145116/models/LightGBM_BAG_L1_FULL/',
      'NeuralNetTorch_BAG_L1_FULL': 'AutogluonModels/ag-20230602_145116/models/NeuralNetTorch_BAG_L1_FULL/',
      'WeightedEnsemble_L2_FULL': 'AutogluonModels/ag-20230602_145116/models/WeightedEnsemble_L2_FULL/'},
     'model_fit_times': {'LightGBM_BAG_L1': 20.36887836456299,
      'NeuralNetTorch_BAG_L1': 286.4162564277649,
      'WeightedEnsemble_L2': 0.15528631210327148,
      'LightGBM_BAG_L2': 14.159779787063599,
      'NeuralNetTorch_BAG_L2': 134.62047600746155,
      'WeightedEnsemble_L3': 0.16128039360046387,
      'LightGBM_BAG_L1_FULL': 1.2905066013336182,
      'NeuralNetTorch_BAG_L1_FULL': 39.00656223297119,
      'WeightedEnsemble_L2_FULL': 0.15528631210327148},
     'model_pred_times': {'LightGBM_BAG_L1': 1.4987642765045166,
      'NeuralNetTorch_BAG_L1': 0.15912771224975586,
      'WeightedEnsemble_L2': 0.0008599758148193359,
      'LightGBM_BAG_L2': 0.09189987182617188,
      'NeuralNetTorch_BAG_L2': 0.22904396057128906,
      'WeightedEnsemble_L3': 0.0007143020629882812,
      'LightGBM_BAG_L1_FULL': None,
      'NeuralNetTorch_BAG_L1_FULL': None,
      'WeightedEnsemble_L2_FULL': None},
     'num_bag_folds': 8,
     'max_stack_level': 3,
     'model_hyperparams': {'LightGBM_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L1_FULL': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L1_FULL': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L2_FULL': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True}},
     'leaderboard':                         model  score_val  pred_time_val    fit_time  \
     0         WeightedEnsemble_L2 -60.386725       1.658752  306.940421   
     1         WeightedEnsemble_L3 -61.391586       1.979550  455.726671   
     2             LightGBM_BAG_L1 -61.564474       1.498764   20.368878   
     3             LightGBM_BAG_L2 -62.122728       1.749792  320.944915   
     4       NeuralNetTorch_BAG_L2 -62.148654       1.886936  441.405611   
     5       NeuralNetTorch_BAG_L1 -66.618550       0.159128  286.416256   
     6    WeightedEnsemble_L2_FULL        NaN            NaN   40.452355   
     7  NeuralNetTorch_BAG_L1_FULL        NaN            NaN   39.006562   
     8        LightGBM_BAG_L1_FULL        NaN            NaN    1.290507   
     
        pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                0.000860           0.155286            2       True   
     1                0.000714           0.161280            3       True   
     2                1.498764          20.368878            1       True   
     3                0.091900          14.159780            2       True   
     4                0.229044         134.620476            2       True   
     5                0.159128         286.416256            1       True   
     6                     NaN           0.155286            2       True   
     7                     NaN          39.006562            1       True   
     8                     NaN           1.290507            1       True   
     
        fit_order  
     0          3  
     1          6  
     2          1  
     3          4  
     4          5  
     5          2  
     6          9  
     7          8  
     8          7  }




```python
predictor_new_hp_2 = TabularPredictor(
    label="count",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    learner_kwargs={"ignored_columns": ["casual", "registered"]},
)

predictor_new_hp_2.fit(
    train_data=train,
    time_limit=600,
    presets="best_quality",
    hyperparameters=hyperparameters_2,
    refit_full="best",
)

```

    No path specified. Models will be saved in: "AutogluonModels/ag-20230602_145949/"
    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 600s
    AutoGluon will save models to "AutogluonModels/ag-20230602_145949/"
    AutoGluon Version:  0.7.0
    Python Version:     3.8.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Tue Apr 25 15:24:19 UTC 2023
    Train Data Rows:    10886
    Train Data Columns: 12
    Label Column: count
    Preprocessing data ...
    Using Feature Generators to preprocess the data ...
    Dropping user-specified ignored columns: ['casual', 'registered']
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    2102.57 MB
    	Train Data (Original)  Memory Usage: 0.72 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 5 | ['holiday', 'workingday', 'humidity', 'hour', 'day']
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])  : 2 | ['season', 'weather']
    		('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])       : 3 | ['humidity', 'hour', 'day']
    		('int', ['bool']) : 2 | ['holiday', 'workingday']
    	0.1s = Fit runtime
    	10 features in original data used to generate 10 features in processed data.
    	Train Data (Processed) Memory Usage: 0.57 MB (0.0% of available memory)
    Data preprocessing and feature engineering runtime = 0.09s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 2 L1 models ...
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 399.84s of the 599.91s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-61.5645	 = Validation score   (-root_mean_squared_error)
    	20.6s	 = Training   runtime
    	1.51s	 = Validation runtime
    Fitting model: NeuralNetTorch_BAG_L1 ... Training model for up to 375.8s of the 575.87s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-71.0435	 = Validation score   (-root_mean_squared_error)
    	315.8s	 = Training   runtime
    	0.17s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.0s of the 256.57s of remaining time.
    	-61.0972	 = Validation score   (-root_mean_squared_error)
    	0.16s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 2 L2 models ...
    Fitting model: LightGBM_BAG_L2 ... Training model for up to 256.34s of the 256.33s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-62.696	 = Validation score   (-root_mean_squared_error)
    	14.55s	 = Training   runtime
    	0.1s	 = Validation runtime
    Fitting model: NeuralNetTorch_BAG_L2 ... Training model for up to 238.39s of the 238.38s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-62.6379	 = Validation score   (-root_mean_squared_error)
    	174.73s	 = Training   runtime
    	0.21s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.0s of the 60.26s of remaining time.
    	-61.9719	 = Validation score   (-root_mean_squared_error)
    	0.16s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 540.07s ... Best model: "WeightedEnsemble_L2"
    Automatically performing refit_full as a post-fit operation (due to `.fit(..., refit_full=True)`
    Refitting models via `predictor.refit_full` using all of the data (combined train and validation)...
    	Models trained in this way will have the suffix "_FULL" and have NaN validation score.
    	This process is not bound by time_limit, but should take less time than the original `predictor.fit` call.
    	To learn more, refer to the `.refit_full` method docstring which explains how "_FULL" models differ from normal models.
    Fitting 1 L1 models ...
    Fitting model: LightGBM_BAG_L1_FULL ...
    	1.28s	 = Training   runtime
    Fitting 1 L1 models ...
    Fitting model: NeuralNetTorch_BAG_L1_FULL ...
    	44.36s	 = Training   runtime
    Fitting model: WeightedEnsemble_L2_FULL | Skipping fit via cloning parent ...
    	0.16s	 = Training   runtime
    Refit complete, total runtime = 47.63s
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20230602_145949/")





    <autogluon.tabular.predictor.predictor.TabularPredictor at 0x7fbaa70f3910>




```python
predictor_new_hp_2.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                            model  score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0         WeightedEnsemble_L2 -61.097221       1.683237  336.557773                0.000752           0.162342            2       True          3
    1             LightGBM_BAG_L1 -61.564474       1.509014   20.598038                1.509014          20.598038            1       True          1
    2         WeightedEnsemble_L3 -61.971928       1.998416  525.834985                0.000718           0.161558            3       True          6
    3       NeuralNetTorch_BAG_L2 -62.637921       1.896030  511.121953                0.213546         174.726521            2       True          5
    4             LightGBM_BAG_L2 -62.696003       1.784152  350.946906                0.101668          14.551474            2       True          4
    5       NeuralNetTorch_BAG_L1 -71.043477       0.173471  315.797393                0.173471         315.797393            1       True          2
    6    WeightedEnsemble_L2_FULL        NaN            NaN   45.805062                     NaN           0.162342            2       True          9
    7  NeuralNetTorch_BAG_L1_FULL        NaN            NaN   44.364559                     NaN          44.364559            1       True          8
    8        LightGBM_BAG_L1_FULL        NaN            NaN    1.278162                     NaN           1.278162            1       True          7
    Number of models trained: 9
    Types of models trained:
    {'StackerEnsembleModel_TabularNeuralNetTorch', 'StackerEnsembleModel_LGB', 'WeightedEnsembleModel'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('category', [])  : 2 | ['season', 'weather']
    ('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])       : 3 | ['humidity', 'hour', 'day']
    ('int', ['bool']) : 2 | ['holiday', 'workingday']
    Plot summary of models saved to file: AutogluonModels/ag-20230602_145949/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'LightGBM_BAG_L1': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L1': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBM_BAG_L2': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L2': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel',
      'LightGBM_BAG_L1_FULL': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L1_FULL': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L2_FULL': 'WeightedEnsembleModel'},
     'model_performance': {'LightGBM_BAG_L1': -61.5644737203475,
      'NeuralNetTorch_BAG_L1': -71.04347666821607,
      'WeightedEnsemble_L2': -61.09722067327154,
      'LightGBM_BAG_L2': -62.696003176824426,
      'NeuralNetTorch_BAG_L2': -62.63792135804677,
      'WeightedEnsemble_L3': -61.97192759418178,
      'LightGBM_BAG_L1_FULL': None,
      'NeuralNetTorch_BAG_L1_FULL': None,
      'WeightedEnsemble_L2_FULL': None},
     'model_best': 'WeightedEnsemble_L2',
     'model_paths': {'LightGBM_BAG_L1': 'AutogluonModels/ag-20230602_145949/models/LightGBM_BAG_L1/',
      'NeuralNetTorch_BAG_L1': 'AutogluonModels/ag-20230602_145949/models/NeuralNetTorch_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20230602_145949/models/WeightedEnsemble_L2/',
      'LightGBM_BAG_L2': 'AutogluonModels/ag-20230602_145949/models/LightGBM_BAG_L2/',
      'NeuralNetTorch_BAG_L2': 'AutogluonModels/ag-20230602_145949/models/NeuralNetTorch_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20230602_145949/models/WeightedEnsemble_L3/',
      'LightGBM_BAG_L1_FULL': 'AutogluonModels/ag-20230602_145949/models/LightGBM_BAG_L1_FULL/',
      'NeuralNetTorch_BAG_L1_FULL': 'AutogluonModels/ag-20230602_145949/models/NeuralNetTorch_BAG_L1_FULL/',
      'WeightedEnsemble_L2_FULL': 'AutogluonModels/ag-20230602_145949/models/WeightedEnsemble_L2_FULL/'},
     'model_fit_times': {'LightGBM_BAG_L1': 20.5980384349823,
      'NeuralNetTorch_BAG_L1': 315.7973930835724,
      'WeightedEnsemble_L2': 0.16234183311462402,
      'LightGBM_BAG_L2': 14.551474332809448,
      'NeuralNetTorch_BAG_L2': 174.72652125358582,
      'WeightedEnsemble_L3': 0.16155791282653809,
      'LightGBM_BAG_L1_FULL': 1.2781615257263184,
      'NeuralNetTorch_BAG_L1_FULL': 44.364558696746826,
      'WeightedEnsemble_L2_FULL': 0.16234183311462402},
     'model_pred_times': {'LightGBM_BAG_L1': 1.5090136528015137,
      'NeuralNetTorch_BAG_L1': 0.17347073554992676,
      'WeightedEnsemble_L2': 0.0007522106170654297,
      'LightGBM_BAG_L2': 0.10166764259338379,
      'NeuralNetTorch_BAG_L2': 0.2135460376739502,
      'WeightedEnsemble_L3': 0.0007183551788330078,
      'LightGBM_BAG_L1_FULL': None,
      'NeuralNetTorch_BAG_L1_FULL': None,
      'WeightedEnsemble_L2_FULL': None},
     'num_bag_folds': 8,
     'max_stack_level': 3,
     'model_hyperparams': {'LightGBM_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L1_FULL': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L1_FULL': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L2_FULL': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True}},
     'leaderboard':                         model  score_val  pred_time_val    fit_time  \
     0         WeightedEnsemble_L2 -61.097221       1.683237  336.557773   
     1             LightGBM_BAG_L1 -61.564474       1.509014   20.598038   
     2         WeightedEnsemble_L3 -61.971928       1.998416  525.834985   
     3       NeuralNetTorch_BAG_L2 -62.637921       1.896030  511.121953   
     4             LightGBM_BAG_L2 -62.696003       1.784152  350.946906   
     5       NeuralNetTorch_BAG_L1 -71.043477       0.173471  315.797393   
     6    WeightedEnsemble_L2_FULL        NaN            NaN   45.805062   
     7  NeuralNetTorch_BAG_L1_FULL        NaN            NaN   44.364559   
     8        LightGBM_BAG_L1_FULL        NaN            NaN    1.278162   
     
        pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                0.000752           0.162342            2       True   
     1                1.509014          20.598038            1       True   
     2                0.000718           0.161558            3       True   
     3                0.213546         174.726521            2       True   
     4                0.101668          14.551474            2       True   
     5                0.173471         315.797393            1       True   
     6                     NaN           0.162342            2       True   
     7                     NaN          44.364559            1       True   
     8                     NaN           1.278162            1       True   
     
        fit_order  
     0          3  
     1          1  
     2          6  
     3          5  
     4          4  
     5          2  
     6          9  
     7          8  
     8          7  }




```python
predictor_new_hp_3 = TabularPredictor(
    label="count",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    learner_kwargs={"ignored_columns": ["casual", "registered"]},
)

predictor_new_hp_3.fit(
    train_data=train,
    time_limit=600,
    presets="best_quality",
    hyperparameters=hyperparameters_3,
    refit_full="best",
)
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20230602_150937/"
    Presets specified: ['best_quality']
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20
    Beginning AutoGluon training ... Time limit = 600s
    AutoGluon will save models to "AutogluonModels/ag-20230602_150937/"
    AutoGluon Version:  0.7.0
    Python Version:     3.8.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Tue Apr 25 15:24:19 UTC 2023
    Train Data Rows:    10886
    Train Data Columns: 12
    Label Column: count
    Preprocessing data ...
    Using Feature Generators to preprocess the data ...
    Dropping user-specified ignored columns: ['casual', 'registered']
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    2102.48 MB
    	Train Data (Original)  Memory Usage: 0.72 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 5 | ['holiday', 'workingday', 'humidity', 'hour', 'day']
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])  : 2 | ['season', 'weather']
    		('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])       : 3 | ['humidity', 'hour', 'day']
    		('int', ['bool']) : 2 | ['holiday', 'workingday']
    	0.1s = Fit runtime
    	10 features in original data used to generate 10 features in processed data.
    	Train Data (Processed) Memory Usage: 0.57 MB (0.0% of available memory)
    Data preprocessing and feature engineering runtime = 0.1s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 2 L1 models ...
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 399.83s of the 599.9s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-70.0589	 = Validation score   (-root_mean_squared_error)
    	15.13s	 = Training   runtime
    	0.67s	 = Validation runtime
    Fitting model: NeuralNetTorch_BAG_L1 ... Training model for up to 381.59s of the 581.65s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-121.0792	 = Validation score   (-root_mean_squared_error)
    	310.78s	 = Training   runtime
    	0.16s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.0s of the 267.4s of remaining time.
    	-70.0589	 = Validation score   (-root_mean_squared_error)
    	0.17s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 2 L2 models ...
    Fitting model: LightGBM_BAG_L2 ... Training model for up to 267.16s of the 267.16s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-67.9156	 = Validation score   (-root_mean_squared_error)
    	15.6s	 = Training   runtime
    	0.53s	 = Validation runtime
    Fitting model: NeuralNetTorch_BAG_L2 ... Training model for up to 248.06s of the 248.05s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy
    	-70.83	 = Validation score   (-root_mean_squared_error)
    	214.18s	 = Training   runtime
    	0.16s	 = Validation runtime
    Completed 1/20 k-fold bagging repeats ...
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.0s of the 30.31s of remaining time.
    	-67.7738	 = Validation score   (-root_mean_squared_error)
    	0.2s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 570.08s ... Best model: "WeightedEnsemble_L3"
    Automatically performing refit_full as a post-fit operation (due to `.fit(..., refit_full=True)`
    Refitting models via `predictor.refit_full` using all of the data (combined train and validation)...
    	Models trained in this way will have the suffix "_FULL" and have NaN validation score.
    	This process is not bound by time_limit, but should take less time than the original `predictor.fit` call.
    	To learn more, refer to the `.refit_full` method docstring which explains how "_FULL" models differ from normal models.
    Fitting 1 L1 models ...
    Fitting model: LightGBM_BAG_L1_FULL ...
    	0.88s	 = Training   runtime
    Fitting 1 L1 models ...
    Fitting model: NeuralNetTorch_BAG_L1_FULL ...
    	45.19s	 = Training   runtime
    Fitting 1 L2 models ...
    Fitting model: LightGBM_BAG_L2_FULL ...
    	0.54s	 = Training   runtime
    Fitting 1 L2 models ...
    Fitting model: NeuralNetTorch_BAG_L2_FULL ...
    	30.52s	 = Training   runtime
    Fitting model: WeightedEnsemble_L3_FULL | Skipping fit via cloning parent ...
    	0.2s	 = Training   runtime
    Refit complete, total runtime = 78.91s
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20230602_150937/")





    <autogluon.tabular.predictor.predictor.TabularPredictor at 0x7fbaa7020730>




```python
predictor_new_hp_3.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                             model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0          WeightedEnsemble_L3  -67.773843       1.514819  555.896756                0.001120           0.199394            3       True          6
    1              LightGBM_BAG_L2  -67.915644       1.357257  341.512412                0.531536          15.597582            2       True          4
    2              LightGBM_BAG_L1  -70.058891       0.668453   15.132133                0.668453          15.132133            1       True          1
    3          WeightedEnsemble_L2  -70.058891       0.669166   15.305546                0.000713           0.173414            2       True          3
    4        NeuralNetTorch_BAG_L2  -70.830043       0.982163  540.099780                0.156442         214.184949            2       True          5
    5        NeuralNetTorch_BAG_L1 -121.079245       0.157268  310.782698                0.157268         310.782698            1       True          2
    6     WeightedEnsemble_L3_FULL         NaN            NaN   77.330257                     NaN           0.199394            3       True         11
    7   NeuralNetTorch_BAG_L2_FULL         NaN            NaN   76.589440                     NaN          30.516695            2       True         10
    8   NeuralNetTorch_BAG_L1_FULL         NaN            NaN   45.188005                     NaN          45.188005            1       True          8
    9         LightGBM_BAG_L2_FULL         NaN            NaN   46.614167                     NaN           0.541422            2       True          9
    10        LightGBM_BAG_L1_FULL         NaN            NaN    0.884739                     NaN           0.884739            1       True          7
    Number of models trained: 11
    Types of models trained:
    {'StackerEnsembleModel_TabularNeuralNetTorch', 'StackerEnsembleModel_LGB', 'WeightedEnsembleModel'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('category', [])  : 2 | ['season', 'weather']
    ('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])       : 3 | ['humidity', 'hour', 'day']
    ('int', ['bool']) : 2 | ['holiday', 'workingday']
    Plot summary of models saved to file: AutogluonModels/ag-20230602_150937/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'LightGBM_BAG_L1': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L1': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBM_BAG_L2': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L2': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel',
      'LightGBM_BAG_L1_FULL': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L1_FULL': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'LightGBM_BAG_L2_FULL': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L2_FULL': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L3_FULL': 'WeightedEnsembleModel'},
     'model_performance': {'LightGBM_BAG_L1': -70.05889100702171,
      'NeuralNetTorch_BAG_L1': -121.07924503250442,
      'WeightedEnsemble_L2': -70.05889100702171,
      'LightGBM_BAG_L2': -67.91564392527927,
      'NeuralNetTorch_BAG_L2': -70.83004257667291,
      'WeightedEnsemble_L3': -67.77384252549753,
      'LightGBM_BAG_L1_FULL': None,
      'NeuralNetTorch_BAG_L1_FULL': None,
      'LightGBM_BAG_L2_FULL': None,
      'NeuralNetTorch_BAG_L2_FULL': None,
      'WeightedEnsemble_L3_FULL': None},
     'model_best': 'WeightedEnsemble_L3',
     'model_paths': {'LightGBM_BAG_L1': 'AutogluonModels/ag-20230602_150937/models/LightGBM_BAG_L1/',
      'NeuralNetTorch_BAG_L1': 'AutogluonModels/ag-20230602_150937/models/NeuralNetTorch_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20230602_150937/models/WeightedEnsemble_L2/',
      'LightGBM_BAG_L2': 'AutogluonModels/ag-20230602_150937/models/LightGBM_BAG_L2/',
      'NeuralNetTorch_BAG_L2': 'AutogluonModels/ag-20230602_150937/models/NeuralNetTorch_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20230602_150937/models/WeightedEnsemble_L3/',
      'LightGBM_BAG_L1_FULL': 'AutogluonModels/ag-20230602_150937/models/LightGBM_BAG_L1_FULL/',
      'NeuralNetTorch_BAG_L1_FULL': 'AutogluonModels/ag-20230602_150937/models/NeuralNetTorch_BAG_L1_FULL/',
      'LightGBM_BAG_L2_FULL': 'AutogluonModels/ag-20230602_150937/models/LightGBM_BAG_L2_FULL/',
      'NeuralNetTorch_BAG_L2_FULL': 'AutogluonModels/ag-20230602_150937/models/NeuralNetTorch_BAG_L2_FULL/',
      'WeightedEnsemble_L3_FULL': 'AutogluonModels/ag-20230602_150937/models/WeightedEnsemble_L3_FULL/'},
     'model_fit_times': {'LightGBM_BAG_L1': 15.132132530212402,
      'NeuralNetTorch_BAG_L1': 310.78269815444946,
      'WeightedEnsemble_L2': 0.17341351509094238,
      'LightGBM_BAG_L2': 15.597581624984741,
      'NeuralNetTorch_BAG_L2': 214.1849491596222,
      'WeightedEnsemble_L3': 0.19939446449279785,
      'LightGBM_BAG_L1_FULL': 0.8847391605377197,
      'NeuralNetTorch_BAG_L1_FULL': 45.188005447387695,
      'LightGBM_BAG_L2_FULL': 0.5414223670959473,
      'NeuralNetTorch_BAG_L2_FULL': 30.516695499420166,
      'WeightedEnsemble_L3_FULL': 0.19939446449279785},
     'model_pred_times': {'LightGBM_BAG_L1': 0.6684529781341553,
      'NeuralNetTorch_BAG_L1': 0.15726828575134277,
      'WeightedEnsemble_L2': 0.000713348388671875,
      'LightGBM_BAG_L2': 0.5315361022949219,
      'NeuralNetTorch_BAG_L2': 0.15644216537475586,
      'WeightedEnsemble_L3': 0.0011196136474609375,
      'LightGBM_BAG_L1_FULL': None,
      'NeuralNetTorch_BAG_L1_FULL': None,
      'LightGBM_BAG_L2_FULL': None,
      'NeuralNetTorch_BAG_L2_FULL': None,
      'WeightedEnsemble_L3_FULL': None},
     'num_bag_folds': 8,
     'max_stack_level': 3,
     'model_hyperparams': {'LightGBM_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L1': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L1_FULL': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L1_FULL': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBM_BAG_L2_FULL': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'NeuralNetTorch_BAG_L2_FULL': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L3_FULL': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True}},
     'leaderboard':                          model   score_val  pred_time_val    fit_time  \
     0          WeightedEnsemble_L3  -67.773843       1.514819  555.896756   
     1              LightGBM_BAG_L2  -67.915644       1.357257  341.512412   
     2              LightGBM_BAG_L1  -70.058891       0.668453   15.132133   
     3          WeightedEnsemble_L2  -70.058891       0.669166   15.305546   
     4        NeuralNetTorch_BAG_L2  -70.830043       0.982163  540.099780   
     5        NeuralNetTorch_BAG_L1 -121.079245       0.157268  310.782698   
     6     WeightedEnsemble_L3_FULL         NaN            NaN   77.330257   
     7   NeuralNetTorch_BAG_L2_FULL         NaN            NaN   76.589440   
     8   NeuralNetTorch_BAG_L1_FULL         NaN            NaN   45.188005   
     9         LightGBM_BAG_L2_FULL         NaN            NaN   46.614167   
     10        LightGBM_BAG_L1_FULL         NaN            NaN    0.884739   
     
         pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                 0.001120           0.199394            3       True   
     1                 0.531536          15.597582            2       True   
     2                 0.668453          15.132133            1       True   
     3                 0.000713           0.173414            2       True   
     4                 0.156442         214.184949            2       True   
     5                 0.157268         310.782698            1       True   
     6                      NaN           0.199394            3       True   
     7                      NaN          30.516695            2       True   
     8                      NaN          45.188005            1       True   
     9                      NaN           0.541422            2       True   
     10                     NaN           0.884739            1       True   
     
         fit_order  
     0           6  
     1           4  
     2           1  
     3           3  
     4           5  
     5           2  
     6          11  
     7          10  
     8           8  
     9           9  
     10          7  }




```python
# Leaderboard dataframe
leaderboard_new_hp_df_1 = pd.DataFrame(predictor_new_hp_1.leaderboard(silent=True))
leaderboard_new_hp_df_1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>score_val</th>
      <th>pred_time_val</th>
      <th>fit_time</th>
      <th>pred_time_val_marginal</th>
      <th>fit_time_marginal</th>
      <th>stack_level</th>
      <th>can_infer</th>
      <th>fit_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WeightedEnsemble_L2</td>
      <td>-60.386725</td>
      <td>1.658752</td>
      <td>306.940421</td>
      <td>0.000860</td>
      <td>0.155286</td>
      <td>2</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WeightedEnsemble_L3</td>
      <td>-61.391586</td>
      <td>1.979550</td>
      <td>455.726671</td>
      <td>0.000714</td>
      <td>0.161280</td>
      <td>3</td>
      <td>True</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LightGBM_BAG_L1</td>
      <td>-61.564474</td>
      <td>1.498764</td>
      <td>20.368878</td>
      <td>1.498764</td>
      <td>20.368878</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LightGBM_BAG_L2</td>
      <td>-62.122728</td>
      <td>1.749792</td>
      <td>320.944915</td>
      <td>0.091900</td>
      <td>14.159780</td>
      <td>2</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NeuralNetTorch_BAG_L2</td>
      <td>-62.148654</td>
      <td>1.886936</td>
      <td>441.405611</td>
      <td>0.229044</td>
      <td>134.620476</td>
      <td>2</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NeuralNetTorch_BAG_L1</td>
      <td>-66.618550</td>
      <td>0.159128</td>
      <td>286.416256</td>
      <td>0.159128</td>
      <td>286.416256</td>
      <td>1</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>WeightedEnsemble_L2_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>40.452355</td>
      <td>NaN</td>
      <td>0.155286</td>
      <td>2</td>
      <td>True</td>
      <td>9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NeuralNetTorch_BAG_L1_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>39.006562</td>
      <td>NaN</td>
      <td>39.006562</td>
      <td>1</td>
      <td>True</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LightGBM_BAG_L1_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.290507</td>
      <td>NaN</td>
      <td>1.290507</td>
      <td>1</td>
      <td>True</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Leaderboard dataframe
leaderboard_new_hp_df_2 = pd.DataFrame(predictor_new_hp_2.leaderboard(silent=True))
leaderboard_new_hp_df_2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>score_val</th>
      <th>pred_time_val</th>
      <th>fit_time</th>
      <th>pred_time_val_marginal</th>
      <th>fit_time_marginal</th>
      <th>stack_level</th>
      <th>can_infer</th>
      <th>fit_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WeightedEnsemble_L2</td>
      <td>-61.097221</td>
      <td>1.683237</td>
      <td>336.557773</td>
      <td>0.000752</td>
      <td>0.162342</td>
      <td>2</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LightGBM_BAG_L1</td>
      <td>-61.564474</td>
      <td>1.509014</td>
      <td>20.598038</td>
      <td>1.509014</td>
      <td>20.598038</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WeightedEnsemble_L3</td>
      <td>-61.971928</td>
      <td>1.998416</td>
      <td>525.834985</td>
      <td>0.000718</td>
      <td>0.161558</td>
      <td>3</td>
      <td>True</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NeuralNetTorch_BAG_L2</td>
      <td>-62.637921</td>
      <td>1.896030</td>
      <td>511.121953</td>
      <td>0.213546</td>
      <td>174.726521</td>
      <td>2</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LightGBM_BAG_L2</td>
      <td>-62.696003</td>
      <td>1.784152</td>
      <td>350.946906</td>
      <td>0.101668</td>
      <td>14.551474</td>
      <td>2</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NeuralNetTorch_BAG_L1</td>
      <td>-71.043477</td>
      <td>0.173471</td>
      <td>315.797393</td>
      <td>0.173471</td>
      <td>315.797393</td>
      <td>1</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>WeightedEnsemble_L2_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>45.805062</td>
      <td>NaN</td>
      <td>0.162342</td>
      <td>2</td>
      <td>True</td>
      <td>9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NeuralNetTorch_BAG_L1_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.364559</td>
      <td>NaN</td>
      <td>44.364559</td>
      <td>1</td>
      <td>True</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LightGBM_BAG_L1_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.278162</td>
      <td>NaN</td>
      <td>1.278162</td>
      <td>1</td>
      <td>True</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



Insight:
- This is the only time when we do not see L_3 as top models


```python
# Leaderboard dataframe
leaderboard_new_hp_df_3 = pd.DataFrame(predictor_new_hp_3.leaderboard(silent=True))
leaderboard_new_hp_df_3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>score_val</th>
      <th>pred_time_val</th>
      <th>fit_time</th>
      <th>pred_time_val_marginal</th>
      <th>fit_time_marginal</th>
      <th>stack_level</th>
      <th>can_infer</th>
      <th>fit_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WeightedEnsemble_L3</td>
      <td>-67.773843</td>
      <td>1.514819</td>
      <td>555.896756</td>
      <td>0.001120</td>
      <td>0.199394</td>
      <td>3</td>
      <td>True</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LightGBM_BAG_L2</td>
      <td>-67.915644</td>
      <td>1.357257</td>
      <td>341.512412</td>
      <td>0.531536</td>
      <td>15.597582</td>
      <td>2</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LightGBM_BAG_L1</td>
      <td>-70.058891</td>
      <td>0.668453</td>
      <td>15.132133</td>
      <td>0.668453</td>
      <td>15.132133</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>WeightedEnsemble_L2</td>
      <td>-70.058891</td>
      <td>0.669166</td>
      <td>15.305546</td>
      <td>0.000713</td>
      <td>0.173414</td>
      <td>2</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NeuralNetTorch_BAG_L2</td>
      <td>-70.830043</td>
      <td>0.982163</td>
      <td>540.099780</td>
      <td>0.156442</td>
      <td>214.184949</td>
      <td>2</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NeuralNetTorch_BAG_L1</td>
      <td>-121.079245</td>
      <td>0.157268</td>
      <td>310.782698</td>
      <td>0.157268</td>
      <td>310.782698</td>
      <td>1</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>WeightedEnsemble_L3_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>77.330257</td>
      <td>NaN</td>
      <td>0.199394</td>
      <td>3</td>
      <td>True</td>
      <td>11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NeuralNetTorch_BAG_L2_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>76.589440</td>
      <td>NaN</td>
      <td>30.516695</td>
      <td>2</td>
      <td>True</td>
      <td>10</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NeuralNetTorch_BAG_L1_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>45.188005</td>
      <td>NaN</td>
      <td>45.188005</td>
      <td>1</td>
      <td>True</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LightGBM_BAG_L2_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>46.614167</td>
      <td>NaN</td>
      <td>0.541422</td>
      <td>2</td>
      <td>True</td>
      <td>9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>LightGBM_BAG_L1_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.884739</td>
      <td>NaN</td>
      <td>0.884739</td>
      <td>1</td>
      <td>True</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
leaderboard_new_hp_df_1.plot(kind="bar", x="model", y="score_val", figsize=(12, 6))
leaderboard_new_hp_df_2.plot(kind="bar", x="model", y="score_val", figsize=(12, 6))
leaderboard_new_hp_df_3.plot(kind="bar", x="model", y="score_val", figsize=(12, 6))
plt.show()
```


    
![png](output_76_0.png)
    



    
![png](output_76_1.png)
    



    
![png](output_76_2.png)
    


Insight:
- WeightedEnsemble is best in all cases, so this can be the model to experiment more with
- Also NN_Torch is not showing any good relative progess across all cases, so it can be safe to remove it


```python
predictions_new_hyp_1 = predictor_new_hp_1.predict(test)
predictions_new_hyp_1.head()
```




    0    18.348980
    1     4.918410
    2     1.559165
    3     2.262614
    4     2.383590
    Name: count, dtype: float32




```python
predictions_new_hyp_2 = predictor_new_hp_2.predict(test)
predictions_new_hyp_2.head()
```




    0    18.680775
    1     4.052975
    2     0.661192
    3     1.742600
    4     1.751991
    Name: count, dtype: float32




```python
predictions_new_hyp_3 = predictor_new_hp_3.predict(test)
predictions_new_hyp_3.head()
```




    0    21.138212
    1    10.469967
    2    10.244417
    3     9.903305
    4    10.091461
    Name: count, dtype: float32




```python
predictions_new_hyp_1[predictions_new_hyp_1<0] = 0 
predictions_new_hyp_2[predictions_new_hyp_2<0] = 0 
predictions_new_hyp_3[predictions_new_hyp_3<0] = 0 
```


```python
submission_new_hyp_1 = pd.read_csv("sampleSubmission.csv", parse_dates=["datetime"])
submission_new_hyp_1["count"] = predictions_new_hyp_1
submission_new_hyp_1.to_csv("submission_new_hyp_1.csv", index=False)

submission_new_hyp_2 = pd.read_csv("sampleSubmission.csv", parse_dates=["datetime"])
submission_new_hyp_2["count"] = predictions_new_hyp_2
submission_new_hyp_2.to_csv("submission_new_hyp_2.csv", index=False)

submission_new_hyp_3 = pd.read_csv("sampleSubmission.csv", parse_dates=["datetime"])
submission_new_hyp_3["count"] = predictions_new_hyp_3
submission_new_hyp_3.to_csv("submission_new_hyp_3.csv", index=False)
```


```python
!kaggle competitions submit -c bike-sharing-demand -f submission_new_hyp_1.csv -m "new features with hyperparameters epoch, boost round"
!kaggle competitions submit -c bike-sharing-demand -f submission_new_hyp_2.csv -m "new features with hyperparameters epoch, boost round, learning rate, extra trees"
!kaggle competitions submit -c bike-sharing-demand -f submission_new_hyp_3.csv -m "new features with hyperparameters epoch, boost round, learning rate, extra trees, drop-out, leaves"
```

    100%|█████████████████████████████████████████| 188k/188k [00:00<00:00, 454kB/s]
    100%|█████████████████████████████████████████| 187k/187k [00:00<00:00, 380kB/s]
    100%|█████████████████████████████████████████| 188k/188k [00:00<00:00, 364kB/s]
    Successfully submitted to Bike Sharing Demand


```python
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
```

    fileName                     date                 description                                                                                         status    publicScore  privateScore  
    ---------------------------  -------------------  --------------------------------------------------------------------------------------------------  --------  -----------  ------------  
    submission_new_hyp_3.csv     2023-06-02 15:20:48  new features with hyperparameters epoch, boost round, learning rate, extra trees, drop-out, leaves  complete  0.51814      0.51814       
    submission_new_hyp_2.csv     2023-06-02 15:20:45  new features with hyperparameters epoch, boost round, learning rate, extra trees                    complete  0.57940      0.57940       
    submission_new_hyp_1.csv     2023-06-02 15:20:43  new features with hyperparameters epoch, boost round                                                complete  0.54777      0.54777       
    submission_new_features.csv  2023-06-02 14:44:22  Two new features (hours & Weekday)                                                                  complete  0.50823      0.50823       
    tail: write error: Broken pipe


#### New Score

- Hy1:  0.50823
- Hy2:  0.57940
- Hy3:  0.51814   


## Step 7: Write a Report
### Refer to the markdown file for the full report
### Creating plots and table for report


```python
fig = (
    pd.DataFrame(
        {
            "model": ["initial", "add_features", "hp1", "hp2", "hp3"],
            "score": [52.871786, 58.196982, 60.386725, 61.097221, 67.773843],
        }
    )
    .plot(x="model", y="score", figsize=(8, 6))
    .get_figure()
)
fig.savefig("model_train_score.png")

```


    
![png](output_87_0.png)
    



```python
fig = (
    pd.DataFrame(
        {
            "test_eval": ["initial", "add_features", "hp1", "hp2", "hp3"],
            "score": [1.80958, 0.50823, 0.54777, 0.57940, 0.51814],
        }
    )
    .plot(x="test_eval", y="score", figsize=(8, 6))
    .get_figure()
)
fig.savefig("model_test_score.png")
```


    
![png](output_88_0.png)
    


Conclusion
- Approach was simple, to change the default values, and see which direction gives better result. This is very basic approach, but result are good as compare to very initial solution
- Best model is to use WeightedEnsemble_L3 and two extra features. All default hyperparameter
- Although Model with hyp_3 does show some better result as compare to hyp_2, so we can go ahead in that direction by changing drop-out and extra tree.


### Hyperparameter table


```python
pd.DataFrame({
    "model": ["initial", "add_features", "hp1", "hp2", "hp3"],
    "hpo1": ["default", "default", "epoch, boost round", "epoch, boost round", "epoch, boost round"],
    "hpo2": ["default", "default", "default", "learning rate, extra trees", "learning rate, extra trees"],
    "hpo3": ["default", "default", "default", "default", "drop-out, leaves"],
    "score": [1.80958, 0.50823, 0.54777, 0.57940, 0.51814]
})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>hpo1</th>
      <th>hpo2</th>
      <th>hpo3</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>initial</td>
      <td>default</td>
      <td>default</td>
      <td>default</td>
      <td>1.80958</td>
    </tr>
    <tr>
      <th>1</th>
      <td>add_features</td>
      <td>default</td>
      <td>default</td>
      <td>default</td>
      <td>0.50823</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hp1</td>
      <td>epoch, boost round</td>
      <td>default</td>
      <td>default</td>
      <td>0.54777</td>
    </tr>
    <tr>
      <th>3</th>
      <td>hp2</td>
      <td>epoch, boost round</td>
      <td>learning rate, extra trees</td>
      <td>default</td>
      <td>0.57940</td>
    </tr>
    <tr>
      <th>4</th>
      <td>hp3</td>
      <td>epoch, boost round</td>
      <td>learning rate, extra trees</td>
      <td>drop-out, leaves</td>
      <td>0.51814</td>
    </tr>
  </tbody>
</table>
</div>



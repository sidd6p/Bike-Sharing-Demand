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
<!-- ![kaggle1.png](attachment:kaggle1.png)
![kaggle2.png](attachment:kaggle2.png) -->
2. Scroll down to API and click Create New API Token.
<!-- ![kaggle3.png](attachment:kaggle3.png)
![kaggle4.png](attachment:kaggle4.png) -->
3. Open up `kaggle.json` and use the username and key.
<!-- ![kaggle5.png](attachment:kaggle5.png) -->

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
kaggle_key = "2e5f7520d0f1ef3e8e133d6b1241caf9"

with open("/root/.kaggle/kaggle.json", "w") as f:
    f.write(json.dumps({"username": kaggle_username, "key": kaggle_key}))
```

### Go to the [bike sharing demand competition](https://www.kaggle.com/c/bike-sharing-demand) and agree to the terms
<!-- ![kaggle6.png](attachment:kaggle6.png) -->


```python
train = pd.read_csv("train.csv",parse_dates=["datetime"])
train.head()
```


```python
train.describe()
```


```python
train.info()
```


```python
test = pd.read_csv("test.csv",parse_dates=["datetime"])
test.head()
```


```python
submission = pd.read_csv("sampleSubmission.csv",parse_dates=["datetime"])
submission.head()
```

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


```python
predictor.fit(train_data=train, time_limit=120, presets="best_quality")
```

### Review AutoGluon's training run with ranking of models that did the best.


```python
predictor.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                       model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0      LightGBMXT_BAG_L2  -61.116628      14.533524  116.015757                3.184328          36.244766            2       True          6
    1    WeightedEnsemble_L3  -61.116628      14.534755  116.020841                0.001231           0.005084            3       True          7
    2  KNeighborsDist_BAG_L1  -84.125061       0.038734    0.029795                0.038734           0.029795            1       True          2
    3    WeightedEnsemble_L2  -84.125061       0.039469    0.285632                0.000736           0.255837            2       True          5
    4  KNeighborsUnif_BAG_L1 -101.546199       0.045319    0.040213                0.045319           0.040213            1       True          1
    5        LightGBM_BAG_L1 -131.077080       1.415203   21.460628                1.415203          21.460628            1       True          4
    6      LightGBMXT_BAG_L1 -131.460909       9.849940   58.240355                9.849940          58.240355            1       True          3
    Number of models trained: 7
    Types of models trained:
    {'WeightedEnsembleModel', 'StackerEnsembleModel_LGB', 'StackerEnsembleModel_KNN'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])                  : 3 | ['season', 'weather', 'humidity']
    ('int', ['bool'])            : 2 | ['holiday', 'workingday']
    ('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    Plot summary of models saved to file: AutogluonModels/ag-20230601_143631/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'KNeighborsUnif_BAG_L1': 'StackerEnsembleModel_KNN',
      'KNeighborsDist_BAG_L1': 'StackerEnsembleModel_KNN',
      'LightGBMXT_BAG_L1': 'StackerEnsembleModel_LGB',
      'LightGBM_BAG_L1': 'StackerEnsembleModel_LGB',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBMXT_BAG_L2': 'StackerEnsembleModel_LGB',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel'},
     'model_performance': {'KNeighborsUnif_BAG_L1': -101.54619908446061,
      'KNeighborsDist_BAG_L1': -84.12506123181602,
      'LightGBMXT_BAG_L1': -131.46090891834504,
      'LightGBM_BAG_L1': -131.0770800258179,
      'WeightedEnsemble_L2': -84.12506123181602,
      'LightGBMXT_BAG_L2': -61.11662760935856,
      'WeightedEnsemble_L3': -61.11662760935856},
     'model_best': 'WeightedEnsemble_L3',
     'model_paths': {'KNeighborsUnif_BAG_L1': 'AutogluonModels/ag-20230601_143631/models/KNeighborsUnif_BAG_L1/',
      'KNeighborsDist_BAG_L1': 'AutogluonModels/ag-20230601_143631/models/KNeighborsDist_BAG_L1/',
      'LightGBMXT_BAG_L1': 'AutogluonModels/ag-20230601_143631/models/LightGBMXT_BAG_L1/',
      'LightGBM_BAG_L1': 'AutogluonModels/ag-20230601_143631/models/LightGBM_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20230601_143631/models/WeightedEnsemble_L2/',
      'LightGBMXT_BAG_L2': 'AutogluonModels/ag-20230601_143631/models/LightGBMXT_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20230601_143631/models/WeightedEnsemble_L3/'},
     'model_fit_times': {'KNeighborsUnif_BAG_L1': 0.04021286964416504,
      'KNeighborsDist_BAG_L1': 0.029795408248901367,
      'LightGBMXT_BAG_L1': 58.240354776382446,
      'LightGBM_BAG_L1': 21.460627794265747,
      'WeightedEnsemble_L2': 0.25583696365356445,
      'LightGBMXT_BAG_L2': 36.244765758514404,
      'WeightedEnsemble_L3': 0.00508427619934082},
     'model_pred_times': {'KNeighborsUnif_BAG_L1': 0.045318603515625,
      'KNeighborsDist_BAG_L1': 0.03873395919799805,
      'LightGBMXT_BAG_L1': 9.849940299987793,
      'LightGBM_BAG_L1': 1.4152026176452637,
      'WeightedEnsemble_L2': 0.0007355213165283203,
      'LightGBMXT_BAG_L2': 3.184328317642212,
      'WeightedEnsemble_L3': 0.0012314319610595703},
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
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'LightGBMXT_BAG_L2': {'use_orig_features': True,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True}},
     'leaderboard':                    model   score_val  pred_time_val    fit_time  \
     0      LightGBMXT_BAG_L2  -61.116628      14.533524  116.015757   
     1    WeightedEnsemble_L3  -61.116628      14.534755  116.020841   
     2  KNeighborsDist_BAG_L1  -84.125061       0.038734    0.029795   
     3    WeightedEnsemble_L2  -84.125061       0.039469    0.285632   
     4  KNeighborsUnif_BAG_L1 -101.546199       0.045319    0.040213   
     5        LightGBM_BAG_L1 -131.077080       1.415203   21.460628   
     6      LightGBMXT_BAG_L1 -131.460909       9.849940   58.240355   
     
        pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                3.184328          36.244766            2       True   
     1                0.001231           0.005084            3       True   
     2                0.038734           0.029795            1       True   
     3                0.000736           0.255837            2       True   
     4                0.045319           0.040213            1       True   
     5                1.415203          21.460628            1       True   
     6                9.849940          58.240355            1       True   
     
        fit_order  
     0          6  
     1          7  
     2          2  
     3          5  
     4          1  
     5          4  
     6          3  }




```python
leaderboard_df = pd.DataFrame(predictor.leaderboard())
leaderboard_df.plot(kind="bar", x="model", y="score_val", figsize=(14, 7))
plt.show()
```

                       model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0      LightGBMXT_BAG_L2  -61.116628      14.533524  116.015757                3.184328          36.244766            2       True          6
    1    WeightedEnsemble_L3  -61.116628      14.534755  116.020841                0.001231           0.005084            3       True          7
    2  KNeighborsDist_BAG_L1  -84.125061       0.038734    0.029795                0.038734           0.029795            1       True          2
    3    WeightedEnsemble_L2  -84.125061       0.039469    0.285632                0.000736           0.255837            2       True          5
    4  KNeighborsUnif_BAG_L1 -101.546199       0.045319    0.040213                0.045319           0.040213            1       True          1
    5        LightGBM_BAG_L1 -131.077080       1.415203   21.460628                1.415203          21.460628            1       True          4
    6      LightGBMXT_BAG_L1 -131.460909       9.849940   58.240355                9.849940          58.240355            1       True          3



    
![png](output_27_1.png)
    


### Create predictions from test dataset


```python
predictions = predictor.predict(test)
predictions.head()
```

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

#### View submission via the command line or in the web browser under the competition's page - `My Submissions`


```python
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
```

#### Initial score of 2.06204     

## Step 4: Exploratory Data Analysis and Creating an additional feature
* Any additional feature will do, but a great suggestion would be to separate out the datetime into hour, day, or month parts.


```python
train.hist(figsize=(15,20))  
plt.tight_layout()
plt.show()
```


    
![png](output_39_0.png)
    



```python
train["hour"] = train["datetime"].dt.hour
train["day"] = train["datetime"].dt.dayofweek
train.drop(["datetime"], axis=1, inplace=True)
train.head()
```


```python
test["hour"] = test["datetime"].dt.hour
test["day"] = test["datetime"].dt.dayofweek
test.drop(["datetime"], axis=1, inplace=True)
test.head()
```

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


```python
train.hist(figsize=(15, 20))
plt.tight_layout()
plt.show()
```


    
![png](output_45_0.png)
    


## Step 5: Rerun the model with the same settings as before, just with more features


```python
predictor_new_features = TabularPredictor(
    label="count",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    learner_kwargs={"ignored_columns": ["casual", "registered"]},
)
```


```python
predictor_new_features.fit(train_data=train, time_limit=120, presets="best_quality")
```


```python
predictor_new_features.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                       model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0    WeightedEnsemble_L3  -59.963984      25.299021  120.690456                0.000761           0.163419            3       True          7
    1    WeightedEnsemble_L2  -60.125100      24.873514   89.367040                0.000833           0.217180            2       True          4
    2      LightGBMXT_BAG_L1  -60.197098      24.770082   89.127638               24.770082          89.127638            1       True          3
    3        LightGBM_BAG_L2  -60.209039      25.080539  104.148664                0.114307          14.976510            2       True          6
    4      LightGBMXT_BAG_L2  -60.977720      25.183953  105.550528                0.217721          16.378374            2       True          5
    5  KNeighborsDist_BAG_L1 -114.004045       0.102599    0.022222                0.102599           0.022222            1       True          2
    6  KNeighborsUnif_BAG_L1 -117.060748       0.093551    0.022294                0.093551           0.022294            1       True          1
    Number of models trained: 7
    Types of models trained:
    {'WeightedEnsembleModel', 'StackerEnsembleModel_LGB', 'StackerEnsembleModel_KNN'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('category', [])  : 2 | ['season', 'weather']
    ('float', [])     : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])       : 3 | ['humidity', 'hour', 'day']
    ('int', ['bool']) : 2 | ['holiday', 'workingday']
    Plot summary of models saved to file: AutogluonModels/ag-20230601_143940/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'KNeighborsUnif_BAG_L1': 'StackerEnsembleModel_KNN',
      'KNeighborsDist_BAG_L1': 'StackerEnsembleModel_KNN',
      'LightGBMXT_BAG_L1': 'StackerEnsembleModel_LGB',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBMXT_BAG_L2': 'StackerEnsembleModel_LGB',
      'LightGBM_BAG_L2': 'StackerEnsembleModel_LGB',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel'},
     'model_performance': {'KNeighborsUnif_BAG_L1': -117.06074757128302,
      'KNeighborsDist_BAG_L1': -114.00404505882429,
      'LightGBMXT_BAG_L1': -60.19709831103628,
      'WeightedEnsemble_L2': -60.12510035164886,
      'LightGBMXT_BAG_L2': -60.977719808009496,
      'LightGBM_BAG_L2': -60.209038712399085,
      'WeightedEnsemble_L3': -59.96398356047644},
     'model_best': 'WeightedEnsemble_L3',
     'model_paths': {'KNeighborsUnif_BAG_L1': 'AutogluonModels/ag-20230601_143940/models/KNeighborsUnif_BAG_L1/',
      'KNeighborsDist_BAG_L1': 'AutogluonModels/ag-20230601_143940/models/KNeighborsDist_BAG_L1/',
      'LightGBMXT_BAG_L1': 'AutogluonModels/ag-20230601_143940/models/LightGBMXT_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20230601_143940/models/WeightedEnsemble_L2/',
      'LightGBMXT_BAG_L2': 'AutogluonModels/ag-20230601_143940/models/LightGBMXT_BAG_L2/',
      'LightGBM_BAG_L2': 'AutogluonModels/ag-20230601_143940/models/LightGBM_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20230601_143940/models/WeightedEnsemble_L3/'},
     'model_fit_times': {'KNeighborsUnif_BAG_L1': 0.022294282913208008,
      'KNeighborsDist_BAG_L1': 0.02222156524658203,
      'LightGBMXT_BAG_L1': 89.12763810157776,
      'WeightedEnsemble_L2': 0.2171802520751953,
      'LightGBMXT_BAG_L2': 16.378373622894287,
      'LightGBM_BAG_L2': 14.97650957107544,
      'WeightedEnsemble_L3': 0.16341924667358398},
     'model_pred_times': {'KNeighborsUnif_BAG_L1': 0.0935513973236084,
      'KNeighborsDist_BAG_L1': 0.1025993824005127,
      'LightGBMXT_BAG_L1': 24.770081520080566,
      'WeightedEnsemble_L2': 0.0008327960968017578,
      'LightGBMXT_BAG_L2': 0.21772050857543945,
      'LightGBM_BAG_L2': 0.11430692672729492,
      'WeightedEnsemble_L3': 0.0007610321044921875},
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
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'max_base_models': 25,
       'max_base_models_per_type': 5,
       'save_bag_folds': True}},
     'leaderboard':                    model   score_val  pred_time_val    fit_time  \
     0    WeightedEnsemble_L3  -59.963984      25.299021  120.690456   
     1    WeightedEnsemble_L2  -60.125100      24.873514   89.367040   
     2      LightGBMXT_BAG_L1  -60.197098      24.770082   89.127638   
     3        LightGBM_BAG_L2  -60.209039      25.080539  104.148664   
     4      LightGBMXT_BAG_L2  -60.977720      25.183953  105.550528   
     5  KNeighborsDist_BAG_L1 -114.004045       0.102599    0.022222   
     6  KNeighborsUnif_BAG_L1 -117.060748       0.093551    0.022294   
     
        pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                0.000761           0.163419            3       True   
     1                0.000833           0.217180            2       True   
     2               24.770082          89.127638            1       True   
     3                0.114307          14.976510            2       True   
     4                0.217721          16.378374            2       True   
     5                0.102599           0.022222            1       True   
     6                0.093551           0.022294            1       True   
     
        fit_order  
     0          7  
     1          4  
     2          3  
     3          6  
     4          5  
     5          2  
     6          1  }




```python
leaderboard_new_df = pd.DataFrame(predictor_new_features.leaderboard())
leaderboard_new_df.plot(kind="bar", x="model", y="score_val", figsize=(14, 7))
plt.show()
```

                       model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0    WeightedEnsemble_L3  -59.963984      25.299021  120.690456                0.000761           0.163419            3       True          7
    1    WeightedEnsemble_L2  -60.125100      24.873514   89.367040                0.000833           0.217180            2       True          4
    2      LightGBMXT_BAG_L1  -60.197098      24.770082   89.127638               24.770082          89.127638            1       True          3
    3        LightGBM_BAG_L2  -60.209039      25.080539  104.148664                0.114307          14.976510            2       True          6
    4      LightGBMXT_BAG_L2  -60.977720      25.183953  105.550528                0.217721          16.378374            2       True          5
    5  KNeighborsDist_BAG_L1 -114.004045       0.102599    0.022222                0.102599           0.022222            1       True          2
    6  KNeighborsUnif_BAG_L1 -117.060748       0.093551    0.022294                0.093551           0.022294            1       True          1



    
![png](output_50_1.png)
    



```python
predictions_new_features = predictor_new_features.predict(test)
predictions_new_features.head()
```


```python
predictions_new_features.describe()
predictions_new_features[predictions_new_features < 0] = 0
```


```python
submission_new_features = pd.read_csv("sampleSubmission.csv", parse_dates=["datetime"])
submission_new_features.head()
```


```python
submission_new_features["count"] = predictions_new_features
submission_new_features.to_csv("submission_new_features.csv", index=False)
```


```python
!kaggle competitions submit -c bike-sharing-demand -f submission_new_features.csv -m "Two new features (hours & Weekday)"
```


```python
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
```

#### New Score of 0.58036      

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
        "learning_rate": 1e-5,
    },
    "GBM": {
        "num_boost_round": 1000,
        "extra_trees": True,
    },
}

```


```python
hyperparameters_3 = {  
    "GBM": {"extra_trees": True, "num_boost_round": 1000, "num_leaves": 5},
    "NN_TORCH": {"num_epochs": 100, "learning_rate": 1e-5, "dropout_prob": 0.05},
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
    time_limit=120,
    presets="best_quality",
    hyperparameters=hyperparameters_1,
    refit_full="best",
)
```


```python
predictor_new_hp_1.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                            model  score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0         WeightedEnsemble_L2 -61.513434       1.707213   81.267027                0.000782           0.166178            2       True          3
    1             LightGBM_BAG_L1 -61.564474       1.525321   20.440489                1.525321          20.440489            1       True          1
    2         WeightedEnsemble_L3 -62.909433       2.188551  123.195870                0.000821           0.168333            3       True          6
    3             LightGBM_BAG_L2 -63.311743       1.793295   95.779210                0.086864          14.678361            2       True          4
    4       NeuralNetTorch_BAG_L2 -65.489339       2.100866  108.349176                0.394434          27.248328            2       True          5
    5       NeuralNetTorch_BAG_L1 -86.110372       0.181111   60.660360                0.181111          60.660360            1       True          2
    6    WeightedEnsemble_L2_FULL        NaN            NaN    8.604759                     NaN           0.166178            2       True          9
    7  NeuralNetTorch_BAG_L1_FULL        NaN            NaN    6.851289                     NaN           6.851289            1       True          8
    8        LightGBM_BAG_L1_FULL        NaN            NaN    1.587292                     NaN           1.587292            1       True          7
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
    Plot summary of models saved to file: AutogluonModels/ag-20230601_144303/SummaryOfModels.html
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
      'NeuralNetTorch_BAG_L1': -86.11037239063127,
      'WeightedEnsemble_L2': -61.51343352900248,
      'LightGBM_BAG_L2': -63.311743480440065,
      'NeuralNetTorch_BAG_L2': -65.4893394423414,
      'WeightedEnsemble_L3': -62.90943306224446,
      'LightGBM_BAG_L1_FULL': None,
      'NeuralNetTorch_BAG_L1_FULL': None,
      'WeightedEnsemble_L2_FULL': None},
     'model_best': 'WeightedEnsemble_L2',
     'model_paths': {'LightGBM_BAG_L1': 'AutogluonModels/ag-20230601_144303/models/LightGBM_BAG_L1/',
      'NeuralNetTorch_BAG_L1': 'AutogluonModels/ag-20230601_144303/models/NeuralNetTorch_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20230601_144303/models/WeightedEnsemble_L2/',
      'LightGBM_BAG_L2': 'AutogluonModels/ag-20230601_144303/models/LightGBM_BAG_L2/',
      'NeuralNetTorch_BAG_L2': 'AutogluonModels/ag-20230601_144303/models/NeuralNetTorch_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20230601_144303/models/WeightedEnsemble_L3/',
      'LightGBM_BAG_L1_FULL': 'AutogluonModels/ag-20230601_144303/models/LightGBM_BAG_L1_FULL/',
      'NeuralNetTorch_BAG_L1_FULL': 'AutogluonModels/ag-20230601_144303/models/NeuralNetTorch_BAG_L1_FULL/',
      'WeightedEnsemble_L2_FULL': 'AutogluonModels/ag-20230601_144303/models/WeightedEnsemble_L2_FULL/'},
     'model_fit_times': {'LightGBM_BAG_L1': 20.440488576889038,
      'NeuralNetTorch_BAG_L1': 60.66035985946655,
      'WeightedEnsemble_L2': 0.16617822647094727,
      'LightGBM_BAG_L2': 14.678361177444458,
      'NeuralNetTorch_BAG_L2': 27.24832773208618,
      'WeightedEnsemble_L3': 0.1683330535888672,
      'LightGBM_BAG_L1_FULL': 1.587292194366455,
      'NeuralNetTorch_BAG_L1_FULL': 6.8512890338897705,
      'WeightedEnsemble_L2_FULL': 0.16617822647094727},
     'model_pred_times': {'LightGBM_BAG_L1': 1.5253205299377441,
      'NeuralNetTorch_BAG_L1': 0.18111109733581543,
      'WeightedEnsemble_L2': 0.0007817745208740234,
      'LightGBM_BAG_L2': 0.08686351776123047,
      'NeuralNetTorch_BAG_L2': 0.39443445205688477,
      'WeightedEnsemble_L3': 0.0008211135864257812,
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
     0         WeightedEnsemble_L2 -61.513434       1.707213   81.267027   
     1             LightGBM_BAG_L1 -61.564474       1.525321   20.440489   
     2         WeightedEnsemble_L3 -62.909433       2.188551  123.195870   
     3             LightGBM_BAG_L2 -63.311743       1.793295   95.779210   
     4       NeuralNetTorch_BAG_L2 -65.489339       2.100866  108.349176   
     5       NeuralNetTorch_BAG_L1 -86.110372       0.181111   60.660360   
     6    WeightedEnsemble_L2_FULL        NaN            NaN    8.604759   
     7  NeuralNetTorch_BAG_L1_FULL        NaN            NaN    6.851289   
     8        LightGBM_BAG_L1_FULL        NaN            NaN    1.587292   
     
        pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                0.000782           0.166178            2       True   
     1                1.525321          20.440489            1       True   
     2                0.000821           0.168333            3       True   
     3                0.086864          14.678361            2       True   
     4                0.394434          27.248328            2       True   
     5                0.181111          60.660360            1       True   
     6                     NaN           0.166178            2       True   
     7                     NaN           6.851289            1       True   
     8                     NaN           1.587292            1       True   
     
        fit_order  
     0          3  
     1          1  
     2          6  
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
    time_limit=120,
    presets="best_quality",
    hyperparameters=hyperparameters_2,
    refit_full="best",
)

```


```python
predictor_new_hp_2.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                       model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0        LightGBM_BAG_L1  -63.345619       2.101660   21.840686                2.101660          21.840686            1       True          1
    1    WeightedEnsemble_L2  -63.345619       2.102354   22.002602                0.000694           0.161916            2       True          3
    2        LightGBM_BAG_L2  -64.039953       3.039567  100.124812                0.795607          18.140151            2       True          4
    3    WeightedEnsemble_L3  -64.039953       3.040397  100.289885                0.000830           0.165072            3       True          6
    4  NeuralNetTorch_BAG_L1 -148.020895       0.142299   60.143975                0.142299          60.143975            1       True          2
    5  NeuralNetTorch_BAG_L2 -320.070707       2.388217  105.632385                0.144257          23.647723            2       True          5
    6   LightGBM_BAG_L1_FULL         NaN            NaN    1.541076                     NaN           1.541076            1       True          7
    Number of models trained: 7
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
    Plot summary of models saved to file: AutogluonModels/ag-20230601_144532/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'LightGBM_BAG_L1': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L1': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBM_BAG_L2': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L2': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel',
      'LightGBM_BAG_L1_FULL': 'StackerEnsembleModel_LGB'},
     'model_performance': {'LightGBM_BAG_L1': -63.345618670334574,
      'NeuralNetTorch_BAG_L1': -148.02089469234645,
      'WeightedEnsemble_L2': -63.345618670334574,
      'LightGBM_BAG_L2': -64.03995275810236,
      'NeuralNetTorch_BAG_L2': -320.0707074589126,
      'WeightedEnsemble_L3': -64.03995275810236,
      'LightGBM_BAG_L1_FULL': None},
     'model_best': 'WeightedEnsemble_L2',
     'model_paths': {'LightGBM_BAG_L1': 'AutogluonModels/ag-20230601_144532/models/LightGBM_BAG_L1/',
      'NeuralNetTorch_BAG_L1': 'AutogluonModels/ag-20230601_144532/models/NeuralNetTorch_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20230601_144532/models/WeightedEnsemble_L2/',
      'LightGBM_BAG_L2': 'AutogluonModels/ag-20230601_144532/models/LightGBM_BAG_L2/',
      'NeuralNetTorch_BAG_L2': 'AutogluonModels/ag-20230601_144532/models/NeuralNetTorch_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20230601_144532/models/WeightedEnsemble_L3/',
      'LightGBM_BAG_L1_FULL': 'AutogluonModels/ag-20230601_144532/models/LightGBM_BAG_L1_FULL/'},
     'model_fit_times': {'LightGBM_BAG_L1': 21.840686082839966,
      'NeuralNetTorch_BAG_L1': 60.143975496292114,
      'WeightedEnsemble_L2': 0.16191577911376953,
      'LightGBM_BAG_L2': 18.140150785446167,
      'NeuralNetTorch_BAG_L2': 23.64772319793701,
      'WeightedEnsemble_L3': 0.16507220268249512,
      'LightGBM_BAG_L1_FULL': 1.5410757064819336},
     'model_pred_times': {'LightGBM_BAG_L1': 2.1016604900360107,
      'NeuralNetTorch_BAG_L1': 0.14229941368103027,
      'WeightedEnsemble_L2': 0.0006935596466064453,
      'LightGBM_BAG_L2': 0.7956070899963379,
      'NeuralNetTorch_BAG_L2': 0.1442568302154541,
      'WeightedEnsemble_L3': 0.0008304119110107422,
      'LightGBM_BAG_L1_FULL': None},
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
       'save_bag_folds': True}},
     'leaderboard':                    model   score_val  pred_time_val    fit_time  \
     0        LightGBM_BAG_L1  -63.345619       2.101660   21.840686   
     1    WeightedEnsemble_L2  -63.345619       2.102354   22.002602   
     2        LightGBM_BAG_L2  -64.039953       3.039567  100.124812   
     3    WeightedEnsemble_L3  -64.039953       3.040397  100.289885   
     4  NeuralNetTorch_BAG_L1 -148.020895       0.142299   60.143975   
     5  NeuralNetTorch_BAG_L2 -320.070707       2.388217  105.632385   
     6   LightGBM_BAG_L1_FULL         NaN            NaN    1.541076   
     
        pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                2.101660          21.840686            1       True   
     1                0.000694           0.161916            2       True   
     2                0.795607          18.140151            2       True   
     3                0.000830           0.165072            3       True   
     4                0.142299          60.143975            1       True   
     5                0.144257          23.647723            2       True   
     6                     NaN           1.541076            1       True   
     
        fit_order  
     0          1  
     1          3  
     2          4  
     3          6  
     4          2  
     5          5  
     6          7  }




```python
predictor_new_hp_3 = TabularPredictor(
    label="count",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    learner_kwargs={"ignored_columns": ["casual", "registered"]},
)

predictor_new_hp_3.fit(
    train_data=train,
    time_limit=120,
    presets="best_quality",
    hyperparameters=hyperparameters_3,
    refit_full="best",
)
```


```python
predictor_new_hp_3.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                            model   score_val  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0             LightGBM_BAG_L2  -68.559370       2.007048   96.484189                0.763687          15.727599            2       True          4
    1         WeightedEnsemble_L3  -68.559370       2.007807   96.651094                0.000760           0.166905            3       True          6
    2             LightGBM_BAG_L1  -72.988425       0.868441   15.433599                0.868441          15.433599            1       True          1
    3         WeightedEnsemble_L2  -72.988425       0.869536   15.625655                0.001095           0.192056            2       True          3
    4       NeuralNetTorch_BAG_L1 -147.026409       0.374919   65.322992                0.374919          65.322992            1       True          2
    5       NeuralNetTorch_BAG_L2 -275.585425       1.372813  107.470856                0.129453          26.714266            2       True          5
    6  NeuralNetTorch_BAG_L1_FULL         NaN            NaN    7.803822                     NaN           7.803822            1       True          8
    7        LightGBM_BAG_L2_FULL         NaN            NaN    9.006895                     NaN           0.599314            2       True          9
    8        LightGBM_BAG_L1_FULL         NaN            NaN    0.603759                     NaN           0.603759            1       True          7
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
    Plot summary of models saved to file: AutogluonModels/ag-20230601_144755/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'LightGBM_BAG_L1': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L1': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBM_BAG_L2': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L2': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel',
      'LightGBM_BAG_L1_FULL': 'StackerEnsembleModel_LGB',
      'NeuralNetTorch_BAG_L1_FULL': 'StackerEnsembleModel_TabularNeuralNetTorch',
      'LightGBM_BAG_L2_FULL': 'StackerEnsembleModel_LGB'},
     'model_performance': {'LightGBM_BAG_L1': -72.98842476114623,
      'NeuralNetTorch_BAG_L1': -147.02640928935065,
      'WeightedEnsemble_L2': -72.98842476114623,
      'LightGBM_BAG_L2': -68.55937007326823,
      'NeuralNetTorch_BAG_L2': -275.58542499426017,
      'WeightedEnsemble_L3': -68.55937007326823,
      'LightGBM_BAG_L1_FULL': None,
      'NeuralNetTorch_BAG_L1_FULL': None,
      'LightGBM_BAG_L2_FULL': None},
     'model_best': 'WeightedEnsemble_L3',
     'model_paths': {'LightGBM_BAG_L1': 'AutogluonModels/ag-20230601_144755/models/LightGBM_BAG_L1/',
      'NeuralNetTorch_BAG_L1': 'AutogluonModels/ag-20230601_144755/models/NeuralNetTorch_BAG_L1/',
      'WeightedEnsemble_L2': 'AutogluonModels/ag-20230601_144755/models/WeightedEnsemble_L2/',
      'LightGBM_BAG_L2': 'AutogluonModels/ag-20230601_144755/models/LightGBM_BAG_L2/',
      'NeuralNetTorch_BAG_L2': 'AutogluonModels/ag-20230601_144755/models/NeuralNetTorch_BAG_L2/',
      'WeightedEnsemble_L3': 'AutogluonModels/ag-20230601_144755/models/WeightedEnsemble_L3/',
      'LightGBM_BAG_L1_FULL': 'AutogluonModels/ag-20230601_144755/models/LightGBM_BAG_L1_FULL/',
      'NeuralNetTorch_BAG_L1_FULL': 'AutogluonModels/ag-20230601_144755/models/NeuralNetTorch_BAG_L1_FULL/',
      'LightGBM_BAG_L2_FULL': 'AutogluonModels/ag-20230601_144755/models/LightGBM_BAG_L2_FULL/'},
     'model_fit_times': {'LightGBM_BAG_L1': 15.433598518371582,
      'NeuralNetTorch_BAG_L1': 65.32299184799194,
      'WeightedEnsemble_L2': 0.19205641746520996,
      'LightGBM_BAG_L2': 15.727598667144775,
      'NeuralNetTorch_BAG_L2': 26.714265823364258,
      'WeightedEnsemble_L3': 0.16690540313720703,
      'LightGBM_BAG_L1_FULL': 0.6037588119506836,
      'NeuralNetTorch_BAG_L1_FULL': 7.803822040557861,
      'LightGBM_BAG_L2_FULL': 0.5993137359619141},
     'model_pred_times': {'LightGBM_BAG_L1': 0.868441104888916,
      'NeuralNetTorch_BAG_L1': 0.37491941452026367,
      'WeightedEnsemble_L2': 0.0010950565338134766,
      'LightGBM_BAG_L2': 0.7636873722076416,
      'NeuralNetTorch_BAG_L2': 0.12945270538330078,
      'WeightedEnsemble_L3': 0.0007596015930175781,
      'LightGBM_BAG_L1_FULL': None,
      'NeuralNetTorch_BAG_L1_FULL': None,
      'LightGBM_BAG_L2_FULL': None},
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
       'save_bag_folds': True}},
     'leaderboard':                         model   score_val  pred_time_val    fit_time  \
     0             LightGBM_BAG_L2  -68.559370       2.007048   96.484189   
     1         WeightedEnsemble_L3  -68.559370       2.007807   96.651094   
     2             LightGBM_BAG_L1  -72.988425       0.868441   15.433599   
     3         WeightedEnsemble_L2  -72.988425       0.869536   15.625655   
     4       NeuralNetTorch_BAG_L1 -147.026409       0.374919   65.322992   
     5       NeuralNetTorch_BAG_L2 -275.585425       1.372813  107.470856   
     6  NeuralNetTorch_BAG_L1_FULL         NaN            NaN    7.803822   
     7        LightGBM_BAG_L2_FULL         NaN            NaN    9.006895   
     8        LightGBM_BAG_L1_FULL         NaN            NaN    0.603759   
     
        pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  \
     0                0.763687          15.727599            2       True   
     1                0.000760           0.166905            3       True   
     2                0.868441          15.433599            1       True   
     3                0.001095           0.192056            2       True   
     4                0.374919          65.322992            1       True   
     5                0.129453          26.714266            2       True   
     6                     NaN           7.803822            1       True   
     7                     NaN           0.599314            2       True   
     8                     NaN           0.603759            1       True   
     
        fit_order  
     0          4  
     1          6  
     2          1  
     3          3  
     4          2  
     5          5  
     6          8  
     7          9  
     8          7  }




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
      <td>-61.513434</td>
      <td>1.707213</td>
      <td>81.267027</td>
      <td>0.000782</td>
      <td>0.166178</td>
      <td>2</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LightGBM_BAG_L1</td>
      <td>-61.564474</td>
      <td>1.525321</td>
      <td>20.440489</td>
      <td>1.525321</td>
      <td>20.440489</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WeightedEnsemble_L3</td>
      <td>-62.909433</td>
      <td>2.188551</td>
      <td>123.195870</td>
      <td>0.000821</td>
      <td>0.168333</td>
      <td>3</td>
      <td>True</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LightGBM_BAG_L2</td>
      <td>-63.311743</td>
      <td>1.793295</td>
      <td>95.779210</td>
      <td>0.086864</td>
      <td>14.678361</td>
      <td>2</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NeuralNetTorch_BAG_L2</td>
      <td>-65.489339</td>
      <td>2.100866</td>
      <td>108.349176</td>
      <td>0.394434</td>
      <td>27.248328</td>
      <td>2</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NeuralNetTorch_BAG_L1</td>
      <td>-86.110372</td>
      <td>0.181111</td>
      <td>60.660360</td>
      <td>0.181111</td>
      <td>60.660360</td>
      <td>1</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>WeightedEnsemble_L2_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.604759</td>
      <td>NaN</td>
      <td>0.166178</td>
      <td>2</td>
      <td>True</td>
      <td>9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NeuralNetTorch_BAG_L1_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.851289</td>
      <td>NaN</td>
      <td>6.851289</td>
      <td>1</td>
      <td>True</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LightGBM_BAG_L1_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.587292</td>
      <td>NaN</td>
      <td>1.587292</td>
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
      <td>LightGBM_BAG_L1</td>
      <td>-63.345619</td>
      <td>2.101660</td>
      <td>21.840686</td>
      <td>2.101660</td>
      <td>21.840686</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WeightedEnsemble_L2</td>
      <td>-63.345619</td>
      <td>2.102354</td>
      <td>22.002602</td>
      <td>0.000694</td>
      <td>0.161916</td>
      <td>2</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LightGBM_BAG_L2</td>
      <td>-64.039953</td>
      <td>3.039567</td>
      <td>100.124812</td>
      <td>0.795607</td>
      <td>18.140151</td>
      <td>2</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>WeightedEnsemble_L3</td>
      <td>-64.039953</td>
      <td>3.040397</td>
      <td>100.289885</td>
      <td>0.000830</td>
      <td>0.165072</td>
      <td>3</td>
      <td>True</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NeuralNetTorch_BAG_L1</td>
      <td>-148.020895</td>
      <td>0.142299</td>
      <td>60.143975</td>
      <td>0.142299</td>
      <td>60.143975</td>
      <td>1</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NeuralNetTorch_BAG_L2</td>
      <td>-320.070707</td>
      <td>2.388217</td>
      <td>105.632385</td>
      <td>0.144257</td>
      <td>23.647723</td>
      <td>2</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LightGBM_BAG_L1_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.541076</td>
      <td>NaN</td>
      <td>1.541076</td>
      <td>1</td>
      <td>True</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




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
      <td>LightGBM_BAG_L2</td>
      <td>-68.559370</td>
      <td>2.007048</td>
      <td>96.484189</td>
      <td>0.763687</td>
      <td>15.727599</td>
      <td>2</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WeightedEnsemble_L3</td>
      <td>-68.559370</td>
      <td>2.007807</td>
      <td>96.651094</td>
      <td>0.000760</td>
      <td>0.166905</td>
      <td>3</td>
      <td>True</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LightGBM_BAG_L1</td>
      <td>-72.988425</td>
      <td>0.868441</td>
      <td>15.433599</td>
      <td>0.868441</td>
      <td>15.433599</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>WeightedEnsemble_L2</td>
      <td>-72.988425</td>
      <td>0.869536</td>
      <td>15.625655</td>
      <td>0.001095</td>
      <td>0.192056</td>
      <td>2</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NeuralNetTorch_BAG_L1</td>
      <td>-147.026409</td>
      <td>0.374919</td>
      <td>65.322992</td>
      <td>0.374919</td>
      <td>65.322992</td>
      <td>1</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NeuralNetTorch_BAG_L2</td>
      <td>-275.585425</td>
      <td>1.372813</td>
      <td>107.470856</td>
      <td>0.129453</td>
      <td>26.714266</td>
      <td>2</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NeuralNetTorch_BAG_L1_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.803822</td>
      <td>NaN</td>
      <td>7.803822</td>
      <td>1</td>
      <td>True</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LightGBM_BAG_L2_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.006895</td>
      <td>NaN</td>
      <td>0.599314</td>
      <td>2</td>
      <td>True</td>
      <td>9</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LightGBM_BAG_L1_FULL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.603759</td>
      <td>NaN</td>
      <td>0.603759</td>
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


    
![png](output_71_0.png)
    



    
![png](output_71_1.png)
    



    
![png](output_71_2.png)
    



```python
predictions_new_hyp_1 = predictor_new_hp_1.predict(test)
predictions_new_hyp_1.head()
```


```python
predictions_new_hyp_2 = predictor_new_hp_2.predict(test)
predictions_new_hyp_2.head()
```


```python
predictions_new_hyp_3 = predictor_new_hp_3.predict(test)
predictions_new_hyp_3.head()
```


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


```python
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
```

#### New Score

- Hy1:  0.60933
- Hy2:  0.65221
- Hy3:  0.50079      

## Step 7: Write a Report
### Refer to the markdown file for the full report
### Creating plots and table for report


```python
fig = (
    pd.DataFrame(
        {
            "model": ["initial", "add_features", "hp1", "hp2", "hp3"],
            "score": [61.018598, 59.963984, 61.523895, 63.345619, 68.541952],
        }
    )
    .plot(x="model", y="score", figsize=(8, 6))
    .get_figure()
)
fig.savefig("model_train_score.png")

```


    
![png](output_81_0.png)
    



```python
fig = (
    pd.DataFrame(
        {
            "test_eval": ["initial", "add_features", "hp1", "hp2", "hp3"],
            "score": [2.08327, 0.58036, 0.73551, 0.65221, 0.50079],
        }
    )
    .plot(x="test_eval", y="score", figsize=(8, 6))
    .get_figure()
)
fig.savefig("model_test_score.png")
```


    
![png](output_82_0.png)
    


### Hyperparameter table


```python
pd.DataFrame({
    "model": ["initial", "add_features", "hp1", "hp2", "hp3"],
    "hpo1": ["default", "default", "epoch, boost round", "epoch, boost round", "epoch, boost round"],
    "hpo2": ["default", "default", "default", "learning rate, extra trees", "learning rate, extra trees"],
    "hpo3": ["default", "default", "default", "default", "drop-out, leaves"],
    "score": [2.08327, 0.58036, 0.73551, 0.65221, 0.50079]
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
      <td>2.08327</td>
    </tr>
    <tr>
      <th>1</th>
      <td>add_features</td>
      <td>default</td>
      <td>default</td>
      <td>default</td>
      <td>0.58036</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hp1</td>
      <td>epoch, boost round</td>
      <td>default</td>
      <td>default</td>
      <td>0.73551</td>
    </tr>
    <tr>
      <th>3</th>
      <td>hp2</td>
      <td>epoch, boost round</td>
      <td>learning rate, extra trees</td>
      <td>default</td>
      <td>0.65221</td>
    </tr>
    <tr>
      <th>4</th>
      <td>hp3</td>
      <td>epoch, boost round</td>
      <td>learning rate, extra trees</td>
      <td>drop-out, leaves</td>
      <td>0.50079</td>
    </tr>
  </tbody>
</table>
</div>



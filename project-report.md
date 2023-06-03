# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Siddhartha Purwar

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
- Need to set negative value to zero, as this is boke sharing prediction and it cannot be negative. Otherwise will get reject by kaggle on submitting it
- Although there was no negtive values in the prediction, so we could safely ignore this

### What was the top ranked model that performed?
- WeightedEnsemble_L3 with add_features (hours and dayofweek)
  - RMSE score of 58.196982
  - Kaggle score of 0.5082.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
- First, there was no null values, so it was easy to deal with data and focus on traing mainly.
- datetime" column in CSV file need to be convert into a datetime object in the resulting dataframe
- In the training dataset we have casual and register column, but now in test dataset. So it does not make any sense to include them in training.

### How much better did your model preform after adding additional features and why do you think that is?
- Score imporved from 1.80958  to 0.5082
- Main reason is that initially datetime was considered as integer. But converting it into datetime object and replacing it with hours and dayofweek, give model two hidden but important feature to train on.
  - There are particular hour of day when people use bike most, like evening and morning. Same way people tend to do biking more on weekends. So it make complete sense to create these two as new feature.
  - Season and weather are categorical variables, so we need to change their type as "category", before that they were considered as just integers.
  
## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
- There were 3 case hyper parameter 
  - hpo1 (score: 0.54777)
  - hpo2 (score: 0.57940)
  - hpo3 (score: 0.51814)
- Perform did not improve as compare to add_feature case, but changing ```drop-out``` and  ```leaves``` can show  better result. As it's score is almost clasoe to add_feature case. 

### If you were given more time with this dataset, where do you think you would spend more time?
- Increse training time
- WeightedEnsemble_L3
- Try adding more feature, like month and year

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
<div>
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



### Create a line plot showing the top model score for the three (or more) training runs during the project.
![image](https://github.com/sidd6p/Bike-Sharing-Demand/assets/91800813/666ca731-d0a9-4c8c-864d-028dec73226d)


### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![image](https://github.com/sidd6p/Bike-Sharing-Demand/assets/91800813/b2e1c8a7-674a-415b-b985-7a585847da8d)    


## Summary
- The AutoGluon AutoML framework make it very easy to run different model and choose the best one.
- The top-ranked model is WeightedEnsemble_L3
- Removing datetime, changing value into correct type helped improve the model a lot
- Increasing training time and changing presets value can be helpful for further better result 

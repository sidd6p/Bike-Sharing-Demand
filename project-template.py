import pandas as pd
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor


## Explore dataset

train = pd.read_csv("data/train.csv", parse_dates=["datetime"])
train.head()
train.describe()
train.info()

test = pd.read_csv("data/test.csv", parse_dates=["datetime"])
test.head()

submission = pd.read_csv("data/sampleSubmission.csv", parse_dates=["datetime"])
submission.head()


## Train a model using AutoGluon’s Tabular Prediction and make prediction

predictor = TabularPredictor(
    label="count",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    learner_kwargs={"ignored_columns": ["casual", "registered"]},
)
predictor.fit(train_data=train, time_limit=600, presets="best_quality")
predictor.fit_summary()

leaderboard_df = pd.DataFrame(predictor.leaderboard())
leaderboard_df.plot(kind="bar", x="model", y="score_val", figsize=(14, 7))
plt.show()

predictions = predictor.predict(test)
predictions[predictions < 0] = 0
predictions.head()

submission["count"] = predictions
submission.to_csv("submission.csv", index=False)

##  Exploratory Data Analysis and Creating an additional feature

train.hist(figsize=(15, 20))
plt.tight_layout()
plt.show()

train["hour"] = train["datetime"].dt.hour
train["day"] = train["datetime"].dt.dayofweek
train.drop(["datetime"], axis=1, inplace=True)
train.head()

test["hour"] = test["datetime"].dt.hour
test["day"] = test["datetime"].dt.dayofweek
test.drop(["datetime"], axis=1, inplace=True)
test.head()

train["season"] = train["season"].astype("category")
train["weather"] = train["weather"].astype("category")

test["season"] = test["season"].astype("category")
test["weather"] = test["weather"].astype("category")

train.hist(figsize=(15, 20))
plt.tight_layout()
plt.show()


##  Re-run the model with the same settings as before, just with more features

predictor_new_features = TabularPredictor(
    label="count",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    learner_kwargs={"ignored_columns": ["casual", "registered"]},
)
predictor_new_features.fit(train_data=train, time_limit=600, presets="best_quality")
predictor_new_features.fit_summary()

leaderboard_new_df = pd.DataFrame(predictor_new_features.leaderboard())
leaderboard_new_df.plot(kind="bar", x="model", y="score_val", figsize=(14, 7))
plt.show()

predictions_new_features = predictor_new_features.predict(test)
predictions_new_features[predictions_new_features < 0] = 0
predictions_new_features.head()
predictions_new_features.describe()

submission_new_features = pd.read_csv("sampleSubmission.csv", parse_dates=["datetime"])
submission_new_features["count"] = predictions_new_features
submission_new_features.to_csv("submission_new_features.csv", index=False)


## Hyper parameter optimization

hyperparameters_1 = {
    "NN_TORCH": {"num_epochs": 100},
    "GBM": {"num_boost_round": 1000},
}
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
    refit_full="best",
)
predictor_new_hp_1.fit_summary()


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
predictor_new_hp_2.fit_summary()

hyperparameters_3 = {
    "GBM": {"extra_trees": True, "num_boost_round": 1000, "num_leaves": 5},
    "NN_TORCH": {"num_epochs": 100, "learning_rate": 1e-5, "dropout_prob": 0.05},
}
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
predictor_new_hp_3.fit_summary()


##  Leaderboard
leaderboard_new_hp_df_1 = pd.DataFrame(predictor_new_hp_1.leaderboard(silent=True))
leaderboard_new_hp_df_1

leaderboard_new_hp_df_2 = pd.DataFrame(predictor_new_hp_2.leaderboard(silent=True))
leaderboard_new_hp_df_2

leaderboard_new_hp_df_3 = pd.DataFrame(predictor_new_hp_3.leaderboard(silent=True))
leaderboard_new_hp_df_3

leaderboard_new_hp_df_1.plot(kind="bar", x="model", y="score_val", figsize=(12, 6))
leaderboard_new_hp_df_2.plot(kind="bar", x="model", y="score_val", figsize=(12, 6))
leaderboard_new_hp_df_3.plot(kind="bar", x="model", y="score_val", figsize=(12, 6))
plt.show()


## Prediction and submission for hypermeter optimisation

predictions_new_hyp_1 = predictor_new_hp_1.predict(test)
predictions_new_hyp_1.head()

predictions_new_hyp_2 = predictor_new_hp_2.predict(test)
predictions_new_hyp_2.head()

predictions_new_hyp_3 = predictor_new_hp_3.predict(test)
predictions_new_hyp_3.head()

predictions_new_hyp_1[predictions_new_hyp_1 < 0] = 0
predictions_new_hyp_2[predictions_new_hyp_2 < 0] = 0
predictions_new_hyp_3[predictions_new_hyp_3 < 0] = 0

submission_new_hyp_1 = pd.read_csv("sampleSubmission.csv", parse_dates=["datetime"])
submission_new_hyp_1["count"] = predictions_new_hyp_1
submission_new_hyp_1.to_csv("submission_new_hyp_1.csv", index=False)

submission_new_hyp_2 = pd.read_csv("sampleSubmission.csv", parse_dates=["datetime"])
submission_new_hyp_2["count"] = predictions_new_hyp_2
submission_new_hyp_2.to_csv("submission_new_hyp_2.csv", index=False)

submission_new_hyp_3 = pd.read_csv("sampleSubmission.csv", parse_dates=["datetime"])
submission_new_hyp_3["count"] = predictions_new_hyp_3
submission_new_hyp_3.to_csv("submission_new_hyp_3.csv", index=False)


## Write a Report

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

pd.DataFrame(
    {
        "model": ["initial", "add_features", "hp1", "hp2", "hp3"],
        "hpo1": [
            "default",
            "default",
            "epoch, boost round",
            "epoch, boost round",
            "epoch, boost round",
        ],
        "hpo2": [
            "default",
            "default",
            "default",
            "learning rate, extra trees",
            "learning rate, extra trees",
        ],
        "hpo3": ["default", "default", "default", "default", "drop-out, leaves"],
        "score": [2.08327, 0.58036, 0.73551, 0.65221, 0.50079],
    }
)

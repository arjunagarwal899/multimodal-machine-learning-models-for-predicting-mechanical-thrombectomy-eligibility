import pandas as pd
import shap
import xgboost
from IPython.display import display

shap.initjs()


def train_xgboost(
    train_x,
    train_y,
    learning_rate,
    n_estimators,
    min_child_weight=2,
    max_depth=3,
    subsample=0.6,
    colsample_bytree=0.6,
    random_state=42,  # Because 42 is the answer to life, the universe and everything
):
    # Initialize the XGBoost model
    model = xgboost.XGBClassifier(
        objective="binary:logistic",
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_weight=min_child_weight,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
    )

    # Fit the model on the training data
    model.fit(train_x, train_y)

    return model


def predict_xgboost(model: xgboost.XGBClassifier, test_x, threshold):
    test_y_score = model.predict_proba(test_x)[:, 1]
    test_y_pred = test_y_score >= threshold
    return test_y_score.tolist(), test_y_pred.tolist()


def get_feature_importances(model, train_x):
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": train_x.columns, "Importance": feature_importances})
    return importance_df


def shap_analysis(model, test_x):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_x)

    print("Explanation of the first 3 predictions:")
    for i in range(3):
        display(shap.force_plot(explainer.expected_value, shap_values[i, :], test_x.iloc[i, :]))
    print("--------------------------------")
    print("Visualization of all test set predictions:")
    display(shap.force_plot(explainer.expected_value, shap_values, test_x))
    print("----------------------------------")
    print("Summary plot:")
    shap.summary_plot(shap_values, test_x, plot_type="bar")
    print("----------------------------------")

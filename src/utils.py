import os
import sys
import dill
import pickle

from sklearn.metrics import r2_score
from src.exception import CustomException


# ✅ SAVE OBJECT
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# ✅ LOAD OBJECT (🔥 REQUIRED FIX)
def load_object(file_path):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")

        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# ✅ SIMPLE EVALUATE MODELS (NO TUNING)
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items():

            model.fit(X_train, y_train)

            # Predictions
            y_test_pred = model.predict(X_test)

            # Score
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
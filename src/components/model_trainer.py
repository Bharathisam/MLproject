import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object   # ❌ removed evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")

            # ✅ Split
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # ✅ Models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(verbosity=0),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
            }

            # ✅ Params (ONLY ONE)
            params = {
                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10],
                },
                "Decision Tree": {
                    "max_depth": [None, 10, 20],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100],
                },
                "Linear Regression": {},
                "KNN": {
                    "n_neighbors": [3, 5]
                },
                "XGBoost": {
                    "n_estimators": [100],
                    "max_depth": [3, 5]
                },
                "CatBoost": {
                    "iterations": [100],
                    "depth": [4, 6]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100]
                }
            }

            # ✅ Hyperparameter tuning loop (FIXED INDENTATION)
            model_report = {}
            best_models = {}

            for model_name, model in models.items():
                param = params[model_name]

                if param:
                    gs = GridSearchCV(
                        model,
                        param,
                        cv=3,
                        scoring="r2",
                        n_jobs=-1
                    )
                    gs.fit(X_train, y_train)
                    best_model = gs.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    best_model = model

                y_pred = best_model.predict(X_test)
                score = r2_score(y_test, y_pred)

                model_report[model_name] = score
                best_models[model_name] = best_model

            # ✅ Best model selection
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = best_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No good model found", sys)

            logging.info(f"Best model: {best_model_name}, Score: {best_model_score}")

            # ✅ Save
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # ✅ Final score
            predicted = best_model.predict(X_test)
            final_r2 = r2_score(y_test, predicted)

            return final_r2

        except Exception as e:
            raise CustomException(e, sys)

        


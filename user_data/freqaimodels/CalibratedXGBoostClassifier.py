import logging
from typing import Any, Dict

from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from freqtrade.freqai.prediction_models.XGBoostClassifier import XGBoostClassifier as FreqaiXGBoostClassifier

logger = logging.getLogger(__name__)


class CalibratedXGBoostClassifier(FreqaiXGBoostClassifier):
    """
    Custom FreqAI Classifier implementing Probability Calibration (Isotonic Regression).
    This wraps the standard XGBoost model in a CalibratedClassifierCV to ensure
    output probabilities are reliable and mathematically calibrated, fixing the
    issue where raw AI scores cause overconfidence.
    """

    def fit(self, data_dictionary: Dict, dk: Any, **kwargs) -> Any:
        """
        Fit the XGBoost classifier and calibrate it using Isotonic Regression.
        """
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            logger.warning("No test set available for calibration. Falling back to training set (may overfit).")

        X_train = data_dictionary["train_features"]
        y_train = data_dictionary["train_labels"]
        
        # Use test set for calibration to avoid overfitting the calibrator
        X_test = data_dictionary.get("test_features", X_train)
        y_test = data_dictionary.get("test_labels", y_train)

        # 1. Initialize base XGBoost model
        base_model = XGBClassifier(
            **self.model_training_parameters,
            random_state=self.freqai_info.get("random_state", 42)
        )

        # 2. Apply Probability Calibration (Isotonic Regression)
        # The calibrator fits the Isotonic curve mapping raw outputs to true empirical probabilities.
        # Using cv=3 lets CalibratedClassifierCV handle the splitting and fitting.
        logger.info("Applying Isotonic Regression Probability Calibration to model...")
        self.model = CalibratedClassifierCV(
            estimator=base_model,
            method="isotonic",
            cv=3
        )
        
        # Fit both the base model (via CV) and the calibrator
        self.model.fit(X_train, y_train)
        
        return self.model

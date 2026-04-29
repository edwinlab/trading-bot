import logging
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRegressor

from freqtrade.freqai.prediction_models.XGBoostRegressor import XGBoostRegressor as FreqaiXGBoostRegressor

logger = logging.getLogger(__name__)


class CalibratedXGBoostRegressor(FreqaiXGBoostRegressor):
    """
    Custom FreqAI Regressor implementing:
    1. Probability Calibration (Isotonic Regression)
    2. Regime-Specific Models (Trending vs Ranging)
    """

    def fit(self, data_dictionary: Dict, dk: Any, **kwargs) -> Any:
        """
        Fit two separate XGBoost regressors (Trending & Ranging) based on ADX.
        Calibrate both using Isotonic Regression.
        """
        # We need to bypass super().fit() completely to train two models, 
        # but we must retain FreqAI's expected return object.
        # We will create a "MetaModel" object to hold our splits.
        
        X_train = data_dictionary["train_features"]
        y_train = data_dictionary["train_labels"]
        X_test = data_dictionary.get("test_features", X_train)
        y_test = data_dictionary.get("test_labels", y_train)

        # Identify ADX column for regime splitting
        adx_cols = [c for c in X_train.columns if 'adx' in c.lower()]
        has_adx = len(adx_cols) > 0
        adx_col = adx_cols[0] if has_adx else None

        MIN_SAMPLES = 50
        is_split = False
        
        if has_adx:
            trend_mask = X_train[adx_col] > 25
            if trend_mask.sum() > MIN_SAMPLES and (~trend_mask).sum() > MIN_SAMPLES:
                is_split = True
                logger.info(f"Splitting model into Trending ({trend_mask.sum()} samples) and Ranging ({(~trend_mask).sum()} samples)")

        class MetaModel:
            def predict(self, X):
                raw_preds = np.zeros(len(X))
                
                if self.is_split:
                    adx_vals = X[self.adx_col]
                    trend_mask = adx_vals > 25
                    range_mask = ~trend_mask
                    
                    if trend_mask.sum() > 0:
                        raw_preds[trend_mask] = self.trend_model.predict(X[trend_mask])
                    if range_mask.sum() > 0:
                        raw_preds[range_mask] = self.range_model.predict(X[range_mask])
                else:
                    raw_preds[:] = self.unified_model.predict(X)
                    
                return raw_preds

        meta_model = MetaModel()
        meta_model.is_split = is_split
        meta_model.adx_col = adx_col

        def train_and_calibrate(X_tr, y_tr, X_te, y_te, suffix=""):
            model = XGBRegressor(
                **self.model_training_parameters,
                random_state=self.freqai_info.get("random_state", 42)
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_tr, y_tr), (X_te, y_te)],
                verbose=False
            )
            
            preds = model.predict(X_te)
            
            # Binary targets for calibration
            binary_y_up = np.where(y_te >= 0.005, 1, 0).ravel()
            binary_y_down = np.where(y_te <= -0.005, 1, 0).ravel()

            try:
                calibrator_up = IsotonicRegression(out_of_bounds="clip")
                calibrator_up.fit(preds, binary_y_up)
            except Exception:
                class DummyCalibrator:
                    def predict(self, x): return np.zeros_like(x)
                calibrator_up = DummyCalibrator()

            try:
                calibrator_down = IsotonicRegression(out_of_bounds="clip")
                calibrator_down.fit(-preds, binary_y_down)
            except Exception:
                class DummyCalibrator:
                    def predict(self, x): return np.zeros_like(x)
                calibrator_down = DummyCalibrator()

            model.calibrator_up = calibrator_up
            model.calibrator_down = calibrator_down
            return model

        if is_split:
            trend_mask_train = X_train[adx_col] > 25
            trend_mask_test = X_test[adx_col] > 25
            
            if trend_mask_test.sum() < 5:
                trend_mask_test = trend_mask_train
                X_test_trend, y_test_trend = X_train[trend_mask_train], y_train[trend_mask_train]
            else:
                X_test_trend, y_test_trend = X_test[trend_mask_test], y_test[trend_mask_test]
                
            meta_model.trend_model = train_and_calibrate(
                X_train[trend_mask_train], y_train[trend_mask_train],
                X_test_trend, y_test_trend, "Trending"
            )

            if (~trend_mask_test).sum() < 5:
                X_test_range, y_test_range = X_train[~trend_mask_train], y_train[~trend_mask_train]
            else:
                X_test_range, y_test_range = X_test[~trend_mask_test], y_test[~trend_mask_test]
                
            meta_model.range_model = train_and_calibrate(
                X_train[~trend_mask_train], y_train[~trend_mask_train],
                X_test_range, y_test_range, "Ranging"
            )
        else:
            logger.info("Not enough samples to split into Regime Models. Training one unified model.")
            meta_model.unified_model = train_and_calibrate(X_train, y_train, X_test, y_test)

        self.model = meta_model
        return self.model

    def predict(self, unfiltered_df, dk, **kwargs) -> Tuple[Any, Any]:
        """
        Predict dynamically routing to the correct regime model, then calibrate.
        """
        pred_df, do_preds = super().predict(unfiltered_df, dk, **kwargs)
        
        if len(pred_df) > 0:
            pred_cols = [c for c in pred_df.columns if c.startswith("&") and not c.endswith("_calibrated")]
            if pred_cols:
                pred_col = pred_cols[0]
                raw_preds = pred_df[pred_col].values
                
                # We need to calibrate based on the regime
                prob_up = np.zeros(len(raw_preds))
                prob_down = np.zeros(len(raw_preds))
                
                features = dk.data_dictionary["prediction_features"]
                
                if self.model.is_split:
                    adx_vals = features[self.model.adx_col].values
                    trend_mask = adx_vals > 25
                    range_mask = ~trend_mask
                    
                    if trend_mask.sum() > 0:
                        prob_up[trend_mask] = self.model.trend_model.calibrator_up.predict(raw_preds[trend_mask])
                        prob_down[trend_mask] = self.model.trend_model.calibrator_down.predict(-raw_preds[trend_mask])
                        
                    if range_mask.sum() > 0:
                        prob_up[range_mask] = self.model.range_model.calibrator_up.predict(raw_preds[range_mask])
                        prob_down[range_mask] = self.model.range_model.calibrator_down.predict(-raw_preds[range_mask])
                else:
                    prob_up[:] = self.model.unified_model.calibrator_up.predict(raw_preds)
                    prob_down[:] = self.model.unified_model.calibrator_down.predict(-raw_preds)
                
                pred_df[f"{pred_col}_prob_up"] = prob_up
                pred_df[f"{pred_col}_prob_down"] = prob_down
                
        return pred_df, do_preds

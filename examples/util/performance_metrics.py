import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.metrics import recall_score, precision_score, fbeta_score


def get_performance_metrics(
    measurements: pd.Series,
    forecasts: pd.Series,
    limit: float
) -> tuple[float, float, float, float, float]:
    """
    Classify per day if a peak was present or not and if it was predicted.
    Then calculate some classification metrics.
    Returns (precision, recall, F-score, actual peaks, forecast peaks)
    """
    measurements_copy = measurements.copy()
    forecasts_copy = forecasts.copy()

    first_day = min(measurements_copy.index.min(), forecasts_copy.index.min())
    first_day = datetime(
        year=first_day.year,
        month=first_day.month,
        day=first_day.day,
        tzinfo=first_day.tzinfo,
    )  # reset to 00:00 hours

    # Correct indices
    measurements_copy.index = measurements_copy.index - first_day
    forecasts_copy.index = forecasts_copy.index - first_day

    # Select the maximum / minimum measurement and forecast per day
    if limit > 0:
        measurements_max = measurements_copy.groupby(by=pd.Grouper(freq='D')).idxmax()
        forecasts_max = forecasts_copy.groupby(by=pd.Grouper(freq='D')).idxmax()
        measurements_selection = measurements_max[measurements_max.isna() == False][forecasts_max.isna() == False]
        forecasts_selection = forecasts_max[measurements_max.isna() == False][forecasts_max.isna() == False]
        peak_values_per_day_actual = measurements_copy.loc[measurements_selection]
        peak_values_per_day_forecast = forecasts_copy.loc[forecasts_selection]
        
    else:
        measurements_min = measurements_copy.groupby(by=pd.Grouper(freq='D')).idxmin()
        forecasts_min = forecasts_copy.groupby(by=pd.Grouper(freq='D')).idxmin()
        measurements_selection = measurements_min[measurements_min.isna() == False][forecasts_min.isna() == False]
        forecasts_selection = forecasts_min[measurements_min.isna() == False][forecasts_min.isna() == False]
        peak_values_per_day_actual = measurements_copy.loc[measurements_selection]
        peak_values_per_day_forecast = forecasts_copy.loc[forecasts_selection]

    # Select actual and forecast peaks
    is_peak_actual = peak_values_per_day_actual.apply(is_peak, args=(limit,))
    is_peak_forecast = peak_values_per_day_forecast.apply(is_peak, args=(limit,))

    # Calculate metrics
    recall = recall_score(is_peak_actual, is_peak_forecast, zero_division=np.nan)
    f10 = fbeta_score(is_peak_actual, is_peak_forecast, beta=10)
    precision = precision_score(is_peak_actual, is_peak_forecast, zero_division=np.nan)
    
    return (precision, recall, f10, len(is_peak_actual[is_peak_actual]), len(is_peak_forecast[is_peak_forecast]))


def is_peak(row: float, limit: float):
    if limit > 0:
        return row > limit
    else:
        return row < limit
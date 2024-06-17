"""
Class to perform quantile transformation of date and numeric columns
"""
import warnings
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from dateutil.parser import parse, ParserError


class SingleColumnToQuantileTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for converting pandas Series into quantiles.

    This transformer handles numeric and datetime columns, transforming
    them based on their own distribution into a specified number of quantiles.
    Columns with a single unique value are transformed to quantile 0.

    Parameters:
    -----------
    column : str
        Nae of Series to be transformed.
    n_bins : int, default=100
        Number of quantiles to use for transforming the data.

    Attributes:
    -----------
    col : stri
        Stores Series name
    n_bins : int
        Stores the number of bins for quantization.
    quantile_bins : list
        List to hold the quantile boundaries.
    dtype : str
        Stores the data type of the series
        ('datetime' or 'numeric').

    Methods:
    --------
    fit(X):
        Fits the transformer to the data, calculating quantiles.
    transform(X):
        Transforms the Series into quantiles.

    """

    def __init__(self, col, dtype, n_bins=100):
        # List of column names to be transformed
        self.col = col
        self.n_bins = n_bins

        # the bins that define quantile boundaries. Populated later.
        self.quantile_bins = None
        self.dtype = dtype

    def fit(self, series):
        """
        Fit the transformer to the data calculating quantiles.
        Does not return anything but update self.quantile_bins.

        Parameters:
        -----------
        series : pd.DataFrame
                DataFrame containing the data to fit.
        """

        series = series[series.notna()].copy()

        if series.shape[0] > 0:
            # convert to numpy compatible date format
            if self.dtype == 'datetime':
                series = series.apply(parse)

            # if dtype is numeric no conversion is needed
            elif self.dtype == 'numeric':
                pass

            self.quantile_bins = np.quantile(
                                    series,
                                    np.linspace(0, 1, num=self.n_bins + 1)
                                    )

    def transform(self, series):
        """
        Transform the DataFrame's columns into quantiles
        based on the fitted data.

        Parameters:
        -----------
        series : pd.Series
                Series to transform.

        Returns:
        --------
        series_trans: pd.Series
            Series transformed to its quantile indices.
        """
        series = series.copy()

        # Apply the quantile bins to transform to quantile indices
        not_na_mask = ~series.isna()
        series = series[not_na_mask]

        series_trans = pd.Series(
            [np.nan]*not_na_mask.shape[0], dtype='float'
        )

        # check if all current data is na
        if series.shape[0] > 0:
            if self.dtype == 'datetime':
                series = series.apply(parse)

            quantile_values = np.searchsorted(
                                    self.quantile_bins,
                                    series,
                                    side='left') / (self.n_bins)

            series_trans[not_na_mask] = quantile_values

        return series_trans


class ToQuantileTransformer(BaseEstimator, TransformerMixin):

    """
    Transformer for converting specified DataFrame columns into quantiles.

    This transformer handles numeric and datetime columns, transforming
    them based on their own distribution into a specified number of quantiles.
    Columns with a single unique value are transformed to quantile 0.
    The class can identify and ignore columns that are neither numeric
    nor datetime.


    Parameters:
    -----------
    columns : list of str
        List of column names in the DataFrame to be transformed.
    n_bins : int, default=100
        Number of quantiles to use for transforming the data.

    Attributes:
    -----------
    cols : list of str
        Stores the column names to be transformed.
    n_bins : int
        Stores the number of bins for quantization.
    quantile_bins : dict
        Dictionary to hold the quantile boundaries for each column.
    dtypes : dict
        Dictionary to store the detected data type of each column
        ('datetime' or 'numeric').
    bad_cols : set
        Set of column names that cannot be classified as either numeric
        or datetime.

    Methods:
    --------
    fit(X, y=None):
        Fits the transformer to the data, identifying column types and
        calculating quantiles.
    transform(X):
        Transforms the DataFrame's specified columns into quantiles.

    """

    def __init__(self, columns, n_bins=100):
        # List of column names to be transformed
        self.cols = list(columns)
        self.n_bins = n_bins

        # the following dictionaries and sets are populated during training.
        # the bins that define quantile boundaries
        self.quantile_bins = {}
        # To store infered dtypes of columns (either 'datetime' or 'numeric')
        self.dtypes = {}
        # Keeps track of columns that were neither datetime nor numeric.
        self.bad_cols = []

    def is_datetime(self, ser: pd.Series):
        """
        Check if the Series is of datetime type.

        Parameters:
        -----------
        ser : pd.Series
            Pandas Series to check for datetime type.

        Returns:
        --------
        bool: True if series is datetime, otherwise False.
        """
        is_datetime = False
        try:
            with warnings.catch_warnings():
                ser.apply(parse)
            is_datetime = True
        except (ValueError, ParserError, TypeError):
            pass
        return is_datetime

    def is_numeric(self, ser: pd.Series):
        """
        Check if the Series is numeric.

        Parameters:
        -----------
        ser : pd.Series
            Pandas Series to check for numeric type.

        Returns:
        --------
        bool
            True if series is numeric, otherwise False.
        """
        is_numeric = False
        try:
            with warnings.catch_warnings():
                ser.astype('float')
            is_numeric = True
        except (ValueError, TypeError):
            pass
        return is_numeric

    def check_dtypes(self, X: pd.DataFrame):
        """
        Determine and store the data types of the specified columns in
        the DataFrame.

        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame containing the columns to check.
        """

        for col in self.cols:
            ser = X[col]
            ser = ser.dropna()

            if self.is_datetime(ser):
                self.dtypes[col] = 'datetime'
            elif self.is_numeric(ser):
                self.dtypes[col] = 'numeric'
            else:
                # the column will be dropped from the dataframe
                self.bad_cols.append(col)

    def fit(self, X):
        """
        Fit the transformer to the data by determining data types and
        calculating quantiles.
        Does not return anything.

        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame containing the data to fit.
        y : Ignored
            This parameter is not used but is included for
            compatibility with scikit-learn.

        """
        self.check_dtypes(X)  # assigns col dtype to self.dtypes

        for col, dtype in self.dtypes.items():
            series = X[col]
            series = series[series.notna()]

            if series.shape[0] > 0:
                # convert to numpy compatible date format
                if dtype == 'datetime':
                    series = series.apply(parse)

                # if dtype is numeric no conversion is needed
                elif dtype == 'numeric':
                    pass

                self.quantile_bins[col] = np.quantile(
                                        series,
                                        np.linspace(0, 1, num=self.n_bins + 1)
                                        )

    def transform(self, X):
        """
        Transform the DataFrame's columns into quantiles
        based on the fitted data.

        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame to transform.

        Returns:
        --------
        X_transformed : pd.DataFrame
            DataFrame with specified columns transformed to their
            quantile indices.
        """
        # subselect columns to transform
        X_transformed = X.copy()#X.drop(columns=list(self.bad_cols)).copy()
        X_transformed[self.bad_cols] = np.nan
        # exclude columns where dtype could not be inferred
        X_cols = set(X_transformed.columns)

        # Apply the quantile bins to transform to quantile indices
        for col, quantile_map in self.quantile_bins.items():
            if col in X_cols:
                series = X_transformed[col]
                not_na_mask = ~series.isna()
                series = series[not_na_mask]

                dtype = self.dtypes[col]

                # check if all current data is na
                if series.shape[0] > 0:
                    if dtype == 'datetime':
                        series = series.apply(parse)

                    quantile_values = np.searchsorted(
                                            quantile_map,
                                            series,
                                            side='left') / (self.n_bins)

                    X_transformed[col] = np.nan
                    X_transformed.loc[not_na_mask, col] = quantile_values

                else:
                    X_transformed[col] = pd.Series(
                        [np.nan]*X_transformed.shape[0], dtype='float'
                    )

        return X_transformed

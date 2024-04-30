""" Utility functions for the encoders in encoders.py """
import pandas as pd
from dateutil.parser import parse, ParserError


def is_datetime(ser: pd.Series, codebook_dtype: str):
    """
    Check if the Series is of datetime type.

    Parameters:
    -----------
    ser : pd.Series
        Pandas Series to check for datetime type.
    codebook: pd.DataFrame
        Pandas dataframe with information on variable types

    Returns:
    --------
    bool: True if series is datetime, otherwise False.
    """
    is_date = False

    if codebook_dtype == 'date or time':
        try:
            ser.apply(parse)
            is_date = True
        except (ValueError, ParserError, TypeError):
            pass

    return is_date


def is_numeric(ser: pd.Series, codebook_dtype: str):
    """
    Check if the Series is numeric.

    Parameters:
    -----------
    ser : pd.Series
        Pandas Series to check for numeric type.
    codebook: pd.DataFrame
        Pandas dataframe with information on variable types

    Returns:
    --------
    bool
        True if series is numeric, otherwise False.
    """
    is_num = False

    if codebook_dtype == 'numeric':
        try:
            ser.astype('float')
            is_num = True
        except (ValueError, TypeError):
            pass

    return is_num


def is_categorical(ser: pd.Series, codebook_dtype: str):
    """
    Check if the Series is categorical.

    Parameters:
    -----------
    ser : pd.Series
        Pandas Series to check for numeric type.
    codebook: pd.DataFrame
        Pandas dataframe with information on variable types

    Returns:
    --------
    bool
        True if series is text, otherwise False.
    """
    is_cat = False

    if codebook_dtype == 'categorical':
        try:
            ser.astype('int')
            is_cat = True
        except (ValueError, TypeError):
            pass

    return is_cat


def is_text(ser: pd.Series, codebook_dtype: str):
    """
    Check if the Series is text.

    Parameters:
    -----------
    ser : pd.Series
        Pandas Series to check for numeric type.
    codebook: pd.DataFrame
        Pandas dataframe with information on variable types

    Returns:
    --------
    bool
        True if series is text, otherwise False.
    """
    is_txt = False

    if codebook_dtype == 'response to open-ended question':
        is_txt = True

    return is_txt


def check_dtypes(ser: pd.Series, codebook: pd.DataFrame):
    """
    Determine and store the data types of the specified columns in
    the DataFrame. Both the codebook and build-in checks have to
    agree for a column to be categorized as a specific dtype.

    Parameters:
    -----------
    ser : pd.Series
        Series to check.
    codebook: pd.DataFrame
        Pandas dataframe with information on variable types
    """
    dtype = None

    ser = ser.dropna().copy()

    codebook_dtype = codebook.loc[codebook['var_name'] == ser.name,
                                  'type_var'].values[0]

    if is_categorical(ser, codebook_dtype):
        dtype = 'categorical'
    elif is_numeric(ser, codebook_dtype):
        dtype = 'numeric'
    elif is_datetime(ser, codebook_dtype):
        dtype = 'datetime'
    elif False:  # is_text(ser, codebook_dtype):
        dtype = 'text'

    return dtype

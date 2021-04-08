import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datetime
import numpy as np
import torch


# One hot encoding for day of week
def one_hot_encode_days(df, drop_column=True):
    """
    Function to one-hot encode the ordinal data for the days of a week
    ---------------------
    Inputs:
    df: Must be pandas dataframe
    If drop_column is set to True:
    Column 'Day of Week' with ordinal data is dropped

    ----------
    Outputs:
    df: Modified pandas dataframe
    """

    # No need to include Sunday. All 6 other days = false means it is Sunday.
    df['Monday'] = (df['Day of Week'] == 0).astype(int)
    df['Tuesday'] = (df['Day of Week'] == 1).astype(int)
    df['Wednesday'] = (df['Day of Week'] == 2).astype(int)
    df['Thursday'] = (df['Day of Week'] == 3).astype(int)
    df['Friday'] = (df['Day of Week'] == 4).astype(int)
    df['Saturday'] = (df['Day of Week'] == 5).astype(int)
    # df['Sunday'] = (df['Day of Week'] == 6).astype(int)

    print('------------------------------')
    print("One-hot encoded 'Day of Week'-Column.")
    if drop_column:
        df.drop('Day of Week', axis=1, inplace=True)
        print("'Day of Week'-Column dropped.")
    print('------------------------------')

    return df


# One hot encoding for months
def one_hot_encode_months(df, drop_column=True):
    """
    Function to one-hot encode the ordinal data for the days of a week
    ---------------------
    Inputs:
    df: Must be pandas dataframe
    If drop_column is set to True:
    Column 'Month' with ordinal data is dropped

    ----------
    Outputs:
    df: Modified pandas dataframe
    """

    # No need to include december. All 11 other months = false means it is december.
    df['January'] = (df['Month'] == 1).astype(int)
    df['February'] = (df['Month'] == 2).astype(int)
    df['March'] = (df['Month'] == 3).astype(int)
    df['April'] = (df['Month'] == 4).astype(int)
    df['May'] = (df['Month'] == 5).astype(int)
    df['June'] = (df['Month'] == 6).astype(int)
    df['July'] = (df['Month'] == 7).astype(int)
    df['August'] = (df['Month'] == 8).astype(int)
    df['September'] = (df['Month'] == 9).astype(int)
    df['October'] = (df['Month'] == 10).astype(int)
    df['November'] = (df['Month'] == 11).astype(int)
    # df['December'] = (df['Month'] == 12).astype(int)

    print('------------------------------')
    print("One-hot encoded 'Month'-Column.")
    if drop_column:
        df.drop('Month', axis=1, inplace=True)
        print("'Month'-Column dropped.")
    print('------------------------------')

    return df


# Encode time-information via sin-cos waves
def sin_cos_waves(df, lst_of_freq=['HalfDay', 'Day', 'Week', 'TwoYears'], drop_datetime_columns=True):
    """
    Function to encode selected time-relevant columns via sin-cos waves
    ---------------------
    Inputs:
    df: Must be pandas dataframe
    lst_of_freq: list of frequencies. Possible elements are 'HalfDay', 'Day', 'Week', 'TwoYears'.
    Each element will be transformed into two columns (sin & cos)
    If drop_datetime_columns is set to True:
    All columns storing time-information ('Hour', 'Day of Month', 'Day of Week', 'Month') will be dropped.

    ----------
    Outputs:
    df: Modified pandas dataframe
    """

    date_time = pd.to_datetime(df['Date']) + pd.to_timedelta(df['Hour'], unit='h')
    timestamp_s = date_time.map(datetime.datetime.timestamp)

    half_day = 12 * 60 * 60  # in seconds
    day = 2 * half_day
    week = day * 7
    two_years = (365.2425) * day * 2

    if 'HalfDay' in lst_of_freq:
        df['HalfDay sin'] = np.sin(timestamp_s * (2 * np.pi / half_day))
        df['HalfDay cos'] = np.cos(timestamp_s * (2 * np.pi / half_day))
    if 'Day' in lst_of_freq:
        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    if 'Week' in lst_of_freq:
        df['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
        df['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
    if 'TwoYears' in lst_of_freq:
        df['TwoYears sin'] = np.sin(timestamp_s * (2 * np.pi / two_years))
        df['TwoYears cos'] = np.cos(timestamp_s * (2 * np.pi / two_years))

    print('------------------------------')
    print('Encoded following frequencies via sin-cos waves: ', [c for c in lst_of_freq])

    if drop_datetime_columns:
        df.drop(['Hour', 'Day of Month', 'Day of Week', 'Month'], axis=1, inplace=True)
        print('Dropped following columns: [Hour, Day of Month, Day of Week, Month]')
    print('------------------------------')

    return df


# time-differencing
def time_differencing(df, column: str = 'Ontario Demand', lag: int = 1, remove_first_rows=True):
    """
    Function to apply time-differencing:
    Instead of absolute values, the function computes the difference of a value compared to the previous time-step.
    ---------------------
    Inputs:
    df: Must be pandas dataframe
    column: String specifying the column name for time-differencing.
    lag: int, defines how many time-steps the function goes back to compute difference.
    If remove_first_row is set to True:
    Removes those rows at the beginning of the differenced series that are NaNs

    ----------
    Outputs:
    df: Modified pandas dataframe
    """

    diff = df[column].diff(periods=lag)
    df['Demand Difference'] = diff.values

    print('------------------------------')
    print('Applied time-differencing on column {} with a lag of {}.'.format(column, lag))
    if remove_first_rows:
        df = df.iloc[lag:]
        print('Dropped NaN rows at the beginning of the dataset.')
    print('------------------------------')

    return df


# apply scaling if desired and split data into train, validation, and test set
def split_and_scale(df,
                    scaler,
                    columns_to_scale=['Ontario Demand', 'Weighted Average Temp (C)'],
                    target_column: str = 'Ontario Demand',
                    vali_set_start_date: str = '2019-07-01',
                    test_set_start_date: str = '2020-01-01'):
    """
    Function to scale specific columns and perform the train-validation-test split.
    ---------------------
    Inputs:
    df: Must be pandas dataframe
    scaler: type of scaler (None, 'min-max', 'standard').
    columns to scale: list, defining the names of columns to be scaled.
    target_column, : str, name of the target column (required for separate scaler).
    If remove_first_row is set to True:
    vali_set_start_date: str, day that defines start of validation set (e.g. '2019-07-01')
    test_set_start_date: str, day that defines start of test set (e.g. '2020-01-01')

    ----------
    Outputs:
    train_df: pandas dataframe storing the training data.
    vali_df: pandas dataframe storing the validation data.
    test_df: pandas dataframe storing the test data.
    target_scaler: fitted scaler (if applicable), possibly required for unscaling
    """

    target_scaler = None

    if scaler is not None:
        if scaler == 'min-max':
            scaler = MinMaxScaler()
            target_scaler = MinMaxScaler()
        elif scaler == 'standard':
            scaler = StandardScaler()
            target_scaler = StandardScaler()
        else:
            raise Exception("Unknown scaler! Possible options are None, 'min-max', and 'standard'.")

        # Fit scaler on training data only
        n_train = df[df.Date == vali_set_start_date].index[0]
        feature_array = df[columns_to_scale].values
        scaler.fit(feature_array[:n_train])

        # Fit target scaler to allow unscaling after model training
        target_array = df[target_column].values
        target_scaler.fit(target_array[:n_train].reshape(-1, 1))

        # Apply scaling on whole dataset and replace unscaled columns in dataframe by scaled ones
        scaled_array = pd.DataFrame(scaler.transform(feature_array), columns=columns_to_scale)
        for column in columns_to_scale:
            df[column] = scaled_array[column]

    # Split dataset
    train_df = df.iloc[0:df[df.Date == vali_set_start_date].index[0]]
    vali_df = df.iloc[df[df.Date == vali_set_start_date].index[0]:df[df.Date == test_set_start_date].index[0]]
    test_df = df.iloc[df[df.Date == test_set_start_date].index[0]:]

    print('------------------------------')
    print('Dataset scaled (if applicable) and split into:')
    print('Training data: {} rows'.format(train_df.shape[0]))
    print('Validation data: {} rows'.format(vali_df.shape[0]))
    print('Test data: {} rows'.format(test_df.shape[0]))
    print('------------------------------')

    return train_df, vali_df, test_df, target_scaler


# apply scaling if desired and split data into train and test set
def split_and_scale_no_val(df,
                          scaler,
                          columns_to_scale=['Ontario Demand', 'Weighted Average Temp (C)'],
                          target_column: str = 'Ontario Demand',
                          test_set_start_date: str = '2020-01-01'):
    """
    Function to scale specific columns and perform the train-test split.
    ---------------------
    Inputs:
    df: Must be pandas dataframe
    scaler: type of scaler (None, 'min-max', 'standard').
    columns to scale: list, defining the names of columns to be scaled.
    target_column, : str, name of the target column (required for separate scaler).
    If remove_first_row is set to True:
    test_set_start_date: str, day that defines start of test set (e.g. '2020-01-01')

    ----------
    Outputs:
    train_df: pandas dataframe storing the training data.
    test_df: pandas dataframe storing the test data.
    target_scaler: fitted scaler (if applicable), possibly required for unscaling
    """

    target_scaler = None

    if scaler is not None:
        if scaler == 'min-max':
            scaler = MinMaxScaler()
            target_scaler = MinMaxScaler()
        elif scaler == 'standard':
            scaler = StandardScaler()
            target_scaler = StandardScaler()
        else:
            raise Exception("Unknown scaler! Possible options are None, 'min-max', and 'standard'.")

        # Fit scaler on training data only
        n_train = df[df.Date == test_set_start_date].index[0]
        feature_array = df[columns_to_scale].values
        scaler.fit(feature_array[:n_train])

        # Fit target scaler to allow unscaling after model training
        target_array = df[target_column].values
        target_scaler.fit(target_array[:n_train].reshape(-1, 1))

        # Apply scaling on whole dataset and replace unscaled columns in dataframe by scaled ones
        scaled_array = pd.DataFrame(scaler.transform(feature_array), columns=columns_to_scale)
        for column in columns_to_scale:
            df[column] = scaled_array[column]

    # Split dataset
    train_df = df.iloc[0:df[df.Date == test_set_start_date].index[0]].copy()
    test_df = df.iloc[df[df.Date == test_set_start_date].index[0]:].copy()

    print('------------------------------')
    print('Dataset scaled (if applicable) and split into:')
    print('Training data: {} rows'.format(train_df.shape[0]))
    print('Test data: {} rows'.format(test_df.shape[0]))
    print('------------------------------')

    return train_df, test_df, target_scaler



# Create sliding windows
def sliding_windows(df, window_size: int = 5, target_column='Ontario Demand', flatten=False,
                    output_window_size: int = 1, perform_feature_shift=False, autoregressive=False):
    """
    Function to expand features to a sliding window
    ---------------------

    Inputs:
    df: Must be pandas dataframe.
    window_size: integer to specify the size of the sliding window
    target columns: string specifying the dependent variable
    If flatten is set to True:
    New number of features = number of features * sliding window size
    New number of time steps = number of time steps - sliding window size
    output_window_size: integer that specifies a forward sliding window to predict future value (default will only predict 1 time step)
    If perform_feature_shift is an integer:
    All dependent variables are shifted N time-step into the future, where N is the integer value.
    

    ---------
    Outputs:
    Returns features and labels separately in numpy arrays.
    If flatten is set to False, the feature data structure will be "3 dimensional"
    1st dimension : time step
    2nd dimension : entry within the sliding window
    3rd dimension : feature

    If flatten is set to True, the feature data structures will be 2 dimensional
    1st dimension : time step
    2nd dimension : each feature for each entry within the sliding window

    """
    df_copy = df.copy()
    if output_window_size < 1:
        raise Exception("Output window must be greater or equal than 1")

    # Drop date column (string value)
    if 'Date' in list(df_copy):
        df_copy.drop('Date', axis=1, inplace=True)


    # Put target column to the end of the dataframe
    df_copy[target_column] = df_copy.pop(target_column)

    print('------------------------------')

    # if applicable, shift features one timestep into the future
    if perform_feature_shift:
        # df_copy[target_column] = df_copy[target_column].shift(-1)
        # df_copy = df_copy.dropna()
        df_copy.iloc[:, :-1] = df_copy.iloc[:, :-1].shift(-perform_feature_shift)
        print('Dependent variables shifted one time-step forward.')

    # Continue with numpy array
    data = df_copy.to_numpy()

    if not autoregressive:
        x = []
        y = []
        for i in range(len(data) - window_size - output_window_size + 1):
            feature_offset = i + window_size
            output_offset = feature_offset + output_window_size
            _x = data[i:feature_offset, :].copy()
            _y = data[feature_offset:output_offset, -1:].copy()

            x.append(_x)
            y.append(_y)

        x = np.array(x)
        y = np.array(y)
        print('Sliding windows created using a window size of {} and forecast length of {}.'
          .format(window_size, output_window_size))
    if autoregressive:
        x = []
        y = []
        for i in range(len(data) - window_size - output_window_size + 1):
            feature_offset = i + window_size
            output_offset = feature_offset + output_window_size
            _x = data[i:output_offset, :].copy()
            _y = data[feature_offset:output_offset, -1:].copy()
            _x[feature_offset:output_offset, -1:] = 0 # Make sure the future electricity data won't leak

            x.append(_x)
            y.append(_y)

        print('Autoregressive Sliding windows created using a window size of  \
        {} and forecast length of {}.'
          .format(window_size, output_window_size))
        x = np.array(x)
        y = np.array(y)
    

    # Flatten output vector - This is always done, we don't want this behaviour
    # within if statement below
    y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])

    # Flatten the feature and window dimensions
    if flatten:
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        print('Feature array flattened.')
    print('------------------------------')

    return x, y

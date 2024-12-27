import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from scipy.stats import boxcox
import numpy as np

# Import deepcopy function
from copy import deepcopy
import datetime

def clean_test_set(test_set, standard_scaler, corr_lambda):
    """
    Cleans the test set and applies transformations such as encoding, scaling, and filtering.

    Parameters:
    test_set (DataFrame): The test dataset to clean.
    standard_scaler (StandardScaler): The fitted scaler for standard scaling.
    corr_lambda (dict): Dictionary of Box-Cox lambda values for each column.
    year_filter (int, optional): Minimum release year to filter the dataset.

    Returns:
    DataFrame: The cleaned test dataset.
    """
    # Filter by release year if year_filter is provided
    test_set = test_set[test_set['released year'] <= 2020]

    # Drop unimportant columns
    columns_to_drop = [
        'in_spotify_playlists',
        'in_spotify_charts',
        'in_apple_playlists',
        'in_apple_charts',
        'in_deezer_playlists',
        'in_deezer_charts',
        'in_shazam_charts',
        'instrumentalness_%'
    ]
    test_set = test_set.drop(columns=columns_to_drop)

    # Remove unwanted characters from column names
    test_set.columns = test_set.columns.str.replace('_', ' ')
    test_set.columns = test_set.columns.str.replace('%', 'percentage')

    # Handle missing values
    test_set['key'] = test_set['key'].fillna("unknown")

    # Standardize string columns
    string_columns = test_set.select_dtypes(include=['object']).columns
    for col in string_columns:
        test_set[col] = test_set[col].str.lower().str.strip()

    # Encode 'mode' column
    test_set['mode'] = (test_set['mode']
                        .replace('unknown', 0)
                        .replace('major', 1)
                        .replace('minor', 2))

    all_keys = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b', 'unknown']
    key_to_index = {key: i for i, key in enumerate(all_keys[:-1])}

    # Encode 'key' column
    def encode_key(key):
        if key == 'unknown':
            return 0.0, 0.0, 1.0  # Unknown key encoding
        index = key_to_index[key]
        sine = np.sin(2 * np.pi * index / 12)
        cosine = np.cos(2 * np.pi * index / 12)
        return sine, cosine, 0.0  # Known key encoding
    
    encoded_keys = {key: encode_key(key) for key in all_keys}

    test_set['key sine'] = test_set['key'].apply(lambda x: encoded_keys[x][0])
    test_set['key cosine'] = test_set['key'].apply(lambda x: encoded_keys[x][1])
    test_set['key unknown'] = test_set['key'].apply(lambda x: encoded_keys[x][2])
    test_set.drop(columns=['key'], inplace=True)

    # Create 'released day of week' column
    def get_day_of_week(year, month, day):
        return datetime.datetime(year, month, day).weekday()

    test_set['released day of week'] = test_set.apply(
        lambda row: get_day_of_week(row['released year'], row['released month'], row['released day']), 
        axis=1
        )

    # Standard scale columns
    columns_to_standard_scale = [
        'bpm',
        'danceability percentage',
        'valence percentage',
        'energy percentage'
    ]
    temp_df = pd.DataFrame(standard_scaler.transform(test_set[columns_to_standard_scale]), columns=columns_to_standard_scale)
    test_set.reset_index(drop=True, inplace=True)
    test_set[columns_to_standard_scale] = temp_df

    # Apply Box-Cox transformation
    columns_to_boxcox_transform = [
        'streams',
        'acousticness percentage',
        'liveness percentage',
        'speechiness percentage'
    ]
    for col in columns_to_boxcox_transform:
        test_set[col] = boxcox(test_set[col] + 1, lmbda=corr_lambda[col])

    # Drop unnecessary columns
    test_set = test_set.drop(columns=[
        'bpm', 'energy percentage', 'acousticness percentage', 'liveness percentage', 
        'released year', 'mode', 'key sine'
        ])

    return test_set


def plot_histogram_and_boxplot(data, column):
    """
    This function plots a histogram and a boxplot for a numerical column in a dataset.

    Parameters:
    data (pandas.DataFrame): The dataset to be used.
    column (str): The name of the numerical column to be plotted.

    Returns:
    None
    """
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
    
    data[column].plot(kind='hist', ax=axs[0], title=column)
    data[column].plot(kind='box', ax=axs[1], title=column, vert=False)

    plt.tight_layout()
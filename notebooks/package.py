import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from scipy.stats import boxcox
import numpy as np

# Import deepcopy function
from copy import deepcopy

columns_to_drop = [
    'released_year',
    'released_month',
    'released_day',
    'in_spotify_playlists',
    'in_spotify_charts',
    'in_apple_playlists',
    'in_apple_charts',
    'in_deezer_playlists',
    'in_deezer_charts',
    'in_shazam_charts',
    'instrumentalness_%',
    'speechiness_%',
    'liveness_%',
]

all_keys = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b', 'unknown']
unique_modes = ['major', 'minor']

# KEY ENCODING
# Assign indices to known keys
key_to_index = {key: i for i, key in enumerate(all_keys[:-1])}

# Encode function
def encode_key(key):
    if key == 'unknown':
        return 0.0, 0.0, 1.0  # Unknown key encoding
    index = key_to_index[key]
    sine = np.sin(2 * np.pi * index / 12)
    cosine = np.cos(2 * np.pi * index / 12)
    return sine, cosine, 0.0  # Known key encoding

# Encode all keys
encoded_keys = {key: encode_key(key) for key in all_keys}

# Convert to DataFrame
keys_df = pd.DataFrame(encoded_keys, index=['sine', 'cosine', 'unknown']).sort_index().T

columns_to_standard_scale = [
    'bpm',
    'danceability percentage',
    'valence percentage',
    'energy percentage',
]

columns_to_boxcox_transform = [
    'streams',
    'acousticness percentage',
]

def clean_test_data(
        dataset: pd.DataFrame,
        standard_scaler: StandardScaler,
        corr_lmbda: list) -> pd.DataFrame:
    # Drop unnecessary columns
    dataset = dataset.drop(columns=columns_to_drop)
    
    # Remove the characters from the column names
    dataset.columns = dataset.columns.str.replace('_', ' ')
    dataset.columns = dataset.columns.str.replace('%', 'percentage')

    # Address Inconsistencies
    # For all string columns, standardize the text by converting them into lowercase
    # and remove leading and trailing whitespaces
    string_columns = dataset.select_dtypes(include=['object']).columns

    for col in string_columns:
        dataset[col] = dataset[col].str.lower().str.strip()
    
    # When cleaning the test set, on columns like `key` and `mode`, 
    # if there's a value doesn't exist in the train set, we will replace it with `unknown`.
    dataset['key'] = dataset['key'].fillna('unknown').apply(lambda x: 'unknown' if x not in all_keys else x)
    dataset['mode'] = dataset['mode'].fillna('unknown').apply(lambda x: 'unknown' if x not in unique_modes else x)
    dataset['mode'] = (dataset['mode']
                    .replace('unknown', 0)
                    .replace('major', 1)
                    .replace('minor', 2))
    dataset['key_sine'] = dataset['key'].apply(lambda x: encoded_keys[x][0])
    dataset['key_cosine'] = dataset['key'].apply(lambda x: encoded_keys[x][1])
    dataset['key_unknown'] = dataset['key'].apply(lambda x: encoded_keys[x][2])
    dataset.drop(columns=['key'], inplace=True)

    # Standardize the nearly normally distributed population columns
    # This assumes that you have fit the standard scaler on the training set
    temp_df = pd.DataFrame(standard_scaler
                       .transform(dataset[columns_to_standard_scale]),
                       columns = columns_to_standard_scale)

    # We have to reset the index, so that when matching the index, the values are assigned correctly.
    # If not, the values will be assigned based on the index of the original dataset, leading to some NaN values.
    dataset.reset_index(drop=True, inplace=True) #optional because we don't remove outliers, thus not affecting the index
    dataset[columns_to_standard_scale] = temp_df

    # Apply Box-Cox transformation on right-skewed columns
    for col in columns_to_boxcox_transform:
        dataset[col] = boxcox(dataset[col] + 1, lmbda=corr_lmbda[col])
    
    return dataset
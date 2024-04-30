from encoders import SingleColumnToQuantileTransformer
from utils import check_dtypes

import pandas as pd


path_for_PreFer_folder = '/Users/lmmi/PreFer/'

codebook = pd.read_csv(
    path_for_PreFer_folder + '/codebooks/PreFer_codebook.csv'
)
df = pd.read_csv(
    path_for_PreFer_folder + 'training_data/PreFer_train_data.csv',
    low_memory=False,
)

# create dict encoder for all relevant columns
encoder = {}
columns = [col for col in df.columns
           if col not in ['nomen_encr', 'outcome_available']]

for col in columns:
    dtype = check_dtypes(ser=df[col], codebook=codebook)

    if dtype == 'categorical':
        pass  # Insert relevant function here

    elif dtype == 'numeric' or dtype == 'datetime':
        trans = SingleColumnToQuantileTransformer(col=col, dtype=dtype)
        trans.fit(series=df[col])
        encoder[col] = trans.transform

    elif dtype == 'text':
        pass  # Insert relevant function here

# example of how to map using the encoder
encoded_df = pd.DataFrame({col: encoder[col](df[col]) for col in encoder})
print(encoded_df)

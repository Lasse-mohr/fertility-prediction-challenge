from os.path import isdir, isfile
from os import makedirs
import pandas as pd
from data_processing.encoding.categorical import CategoricalTransformer
from data_processing.encoding.numeric_and_date import ToQuantileTransformer
from data_processing.sequences.sequencing import to_sequences, get_generic_name


def encoding_pipeline(data, codebook=None, use_codebook=True, custom_pairs=None,
                      save_inter_path='data_processing/codebook_false/encoding_pipeline/',
                      ):

    data = data.copy()

    categorical_column_filepath = save_inter_path + 'categorical_columns.csv'
    quantile_column_filepath = save_inter_path + 'quantile_columns.csv'

    if not use_codebook:
        if isfile(categorical_column_filepath) or not isfile(quantile_column_filepath):
            categorical_columns = pd.read_csv(categorical_column_filepath).squeeze()
            quantile_columns = pd.read_csv(quantile_column_filepath).squeeze()
        else:
            print(f'Files with column names not found at: {categorical_column_filepath}')
            print('Calculating columns with encoding_pipeline')
            use_codebook = True

    if use_codebook:
        # Select only questions with yearly component
        codebook = codebook[codebook.year.notna()]
        # Get all question pairs
        if custom_pairs is not None:
            codebook["pairs"] = codebook['var_name'].apply(get_generic_name)
            codebook = codebook[codebook["pairs"].isin(custom_pairs)]

        # Get relevant columns
        categorical_columns = codebook[codebook.type_var ==
                                       'categorical'].var_name
        quantile_columns = codebook[((codebook.type_var == 'numeric') | (
            codebook.type_var == 'date or time'))].var_name
        # text columns should go here

        if not isdir(save_inter_path):
            makedirs(save_inter_path)

        categorical_columns.to_csv(categorical_column_filepath, index=False)
        quantile_columns.to_csv(quantile_column_filepath, index=False)

    # Encode categorical columns
    categorical_transformer = CategoricalTransformer()
    categorical_transformer.fit(codebook, use_codebook=use_codebook)

    data[categorical_columns] = categorical_transformer.transform(
        data[categorical_columns])

    # Encode numeric and date columns
    quantile_transformer = ToQuantileTransformer(quantile_columns)
    quantile_transformer.fit(data)
    data = quantile_transformer.transform(data)

    # Fill any nans
    data = data.fillna(101)
    data = data.astype(int, errors='ignore')
    # Drop object columns (automatically filled with 101 in to_sequences)
    data = data[data.columns[data.dtypes != 'object']]

    # Convert to sequences
    sequences = to_sequences(data, codebook, use_codebook=use_codebook,
                             custom_pairs=custom_pairs)

    return sequences



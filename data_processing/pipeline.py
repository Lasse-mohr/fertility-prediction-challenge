from data_processing.encoding.categorical import CategoricalTransformer
from data_processing.encoding.numeric_and_date import ToQuantileTransformer
from data_processing.sequences.sequencing import to_sequences


def encoding_pipeline(data, codebook):
    # Select only questions with yearly component
    codebook = codebook[codebook.year.notna()]

    # Get relevant columns
    categorical_columns = codebook[codebook.type_var == 'categorical'].var_name
    quantile_columns = codebook[((codebook.type_var == 'numeric') | (codebook.type_var == 'date or time'))].var_name
    # text columns should go here

    # Encode categorical columns
    categorical_transformer = CategoricalTransformer()
    categorical_transformer.fit(codebook)
    data[categorical_columns] = categorical_transformer.transform(data[categorical_columns])

    # Encode numeric and date columns
    quantile_transformer = ToQuantileTransformer(quantile_columns)
    quantile_transformer.fit(data)
    data = quantile_transformer.transform(data)

    # Encode text columns (SKIPPED)

    # Fill any nans
    data = data.fillna(101)
    data = data.astype(int, errors='ignore')
    data = data[data.columns[data.dtypes != 'object']] # Drop object columns (automatically filled with 101 in to_sequences)

    # Convert to sequences
    sequences = to_sequences(data, codebook)

    return sequences


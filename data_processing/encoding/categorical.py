import re
import pandas as pd
import itertools
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def fit(self, codebook):
        # Get categorical core questions
        core_cat_df = codebook[(codebook.year.notna()) & (codebook.type_var == 'categorical')].copy()

        # Convert strings to list
        core_cat_df['values_cat'] = core_cat_df['values_cat'].str.split("; ").apply(lambda x: [e.strip() for e in x])
        core_cat_df['labels_cat'] = core_cat_df['labels_cat'].str.split("; ").apply(lambda x: [e.strip() for e in x])

        # Init vars
        vocab = {'UNK': 101} # We reserve the first 101 for quantiles
        vocab_max = 102 # (102)
        single_labels = []

        # Group by question_number (first two and last three characters of var_name)
        for question_key, group in core_cat_df.groupby(lambda x: self.get_question_number(core_cat_df['var_name'][x]), sort=False):
            # Get all unique value-label pairs
            pairs = self.get_unique_value_label_pairs(group)
            pairs = sorted(pairs, key=lambda x: x[0]) # Sort by value (needed for itertools.groupby)

            # Iterate over pairs
            for _, subgroup in itertools.groupby(pairs, key=lambda x: x[0]):
                labels = [label for _, label in subgroup]
                # We encode single labels last
                if len(labels) <= 1:    
                    single_labels.append(labels[0])
                    continue
                elif len(labels) >= 2: # When labels changed for a question
                    if self.is_same_label(question_key, labels): # Check if it's rephrasing of the same label
                        # Check for exisiting labels in vocab
                        present_labels = [label for label in labels if label in vocab]
                        # If present, use it, else create new token
                        if len(present_labels) > 0:
                            token_num = vocab[present_labels[0]]
                        else:
                            token_num = vocab_max
                            vocab_max += 1
                        # Add all labels to vocab
                        for label in labels:
                            if label not in vocab:
                                vocab[label] = token_num
                    else: # If labels are not the same, add them to vocab as new tokens
                        for label in labels:
                            if label not in vocab:
                                vocab[label] = vocab_max
                                vocab_max += 1

        # Lastly we add single labels to vocab
        for label in single_labels:
            if label not in vocab:
                vocab[label] = vocab_max
                vocab_max += 1

        # We then create the tokenizer (converter)
        self.converter = {
            row['var_name']: {value: vocab[label] for value, label in zip(row['values_cat'], row['labels_cat'])}
            for _, row in core_cat_df.iterrows()
        }
        
        self.vocab = vocab

    def transform(self, df):
        if isinstance(df, pd.Series):
            return self.transform_series(df)
        elif isinstance(df, pd.DataFrame):
            return df.apply(self.transform_series)
        
    def transform_series(self, series: pd.Series):
        return series.apply(lambda x: self.tokenize(x, self.converter[series.name]))

    def is_same_label(self, question_key, labels):
        lower_labels = [l.lower() for l in labels]
        cleaned_labels = [re.sub(r'[^\w]', ' ', l) for l in lower_labels]
        if self.is_same(cleaned_labels) \
            or self.is_positive_edge_case(question_key, labels) \
            or self.is_value_equal(lower_labels) \
            or self.one_word_match(lower_labels) \
            or self.is_overlapping(cleaned_labels):
            return True
        if self.if_all_digits(labels) or self.is_negative_edge_case(question_key, labels):
            return False
        return False
    
    def tokenize(self, value, series_converter: dict):
        if pd.notna(value):
            value = str(int(value))
        return series_converter.get(value, self.vocab['UNK'])
    
    def max(self):
        return max(self.vocab.values())

    @staticmethod
    def get_question_number(var_name):
        return (var_name[:2], var_name[-3:])

    @staticmethod
    def get_unique_value_label_pairs(group):
        """ Get all unique value-label pairs for a question group"""
        pairs = group.apply(lambda row: list(zip(row['values_cat'], row['labels_cat'])), axis=1)
        pairs = set(sum(pairs.tolist(), []))
        return pairs
    
    @staticmethod
    def if_all_digits(labels): #Also counts dates
        """ Checks whether all labels are digits, should be falsified"""
        return all([l.isdigit() for l in labels])
    @staticmethod
    def is_same(labels):
        """ Checks whether all labels are the same"""
        return len(set(labels)) == 1
    @staticmethod
    def one_word_match(labels):
        """ Checks whether all labels have a word in common"""
        for label in labels:
            for word in label.split():
                if all([word in l.split() for l in labels if l != label]):
                    break
            else:
                return False
        return True
    @staticmethod
    def is_positive_edge_case(question_key, labels):
        """ Positive edge cases manually checked and determined to be the same label"""
        edge_cases = [
            set(['man', 'male']),
            set(['woman', 'female']), 
            set(['definitely no', 'certainly not']),
            set(['yes', 'ja']), 
            set(['nee', 'no']),
            set(['niet van toepassing', 'not applicable']),
        ]
        return set(labels) in edge_cases or question_key == ('cs', '372')
    @staticmethod
    def is_negative_edge_case(question_key, labels): # Should be Falsified (we should negative cases)
        """ Negative edge cases manually checked and should be falsified"""
        edge_cases = [
            set(['less often', 'never']), 
            set(['has paid job', 'yes']),
            set(['Reformed Churches in the Netherlands (Gereformeerd)', 'Hinduism']),
            set(['The traditional version cv17i244', 'The traditional version cv18j244', 'The traditional version cv19k244', 'The traditional version cv20l308']), 
            set(['The chance questions cv20l245 – cv20l262', 'The chance questions cv19k245 – cv19k262', 'The chance questions cv18j245 – cv18j262', 'The chance questions cv17i245 – cv17i262']),
            set(["don't know/don't want to say", 'No', 'no']),
            set(['did not complete any education', 'did not finish primary school'])
        ]
        return set(labels) in edge_cases or question_key == ('cw', '005') or question_key == ('cw', '008')

    @staticmethod
    def is_value_equal(labels):
        """ For the case where a digit is in the labels e.g. (1 = Very good, 1 = Super good)
        Checks whether the values are equal"""
        if not labels[0].split()[0].isdigit():
            return False
        for label in labels:
            if not all([label.split()[0] == l.split()[0] for l in labels]):
                return False
        return True
    @staticmethod
    def is_overlapping(labels):
        """ Checks whether the characters of the labels are overlapping"""
        for label in labels:
            if not all([(set(label).issubset(label2) or set(label2).issubset(label)) for label2 in labels if label2 != label]):
                return False
        return True


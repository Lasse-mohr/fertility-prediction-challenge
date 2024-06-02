# Data packages
import polars as pl     # requires installing polars first
import re

import torch_frame as tf
from collections import defaultdict


# Model
import numpy as np


class CodeBookFilter:
    """
    Class to filter out the codebook
    """

    def __init__(self, path: str, accept_missing_rate: float = 0.05, column_appeared_times: int = 8) -> None:
        assert accept_missing_rate >= 0 and accept_missing_rate <= 1.0
        assert column_appeared_times > 0
        self.column_appeared_times = column_appeared_times
        self.accept_missing_rate = accept_missing_rate

        codebook = pl.read_csv(path).to_pandas()

        self.column_appeared_times = column_appeared_times
        self.codebook = self._filter_codebook(
            codebook=codebook)  # CodeBook Filtered
        self.valid_columns = self._return_valid_columns()
        self._create_column_dict()

    def _filter_codebook(self, codebook):
        codebook = codebook[codebook["prop_missing"]
                            < self.accept_missing_rate]
        codebook = codebook[codebook["type_var"].isin(
            ["numeric", "categorical"])]
        return codebook

    def _return_valid_columns(self):
        column_appeared = defaultdict(int)
        for year in range(2007, 2020):
            _cols = self.codebook[self.codebook["year"]
                                  == year]["var_name"].values
            _cols = [self._match(x) for x in _cols]
            for _c in _cols:
                if _c != None:
                    column_appeared[_c] += 1
        return set([k for k, v in column_appeared.items() if v > self.column_appeared_times])

    def _create_column_dict(self):
        self.col2id = dict()
        self.year2col = dict()
        self.col2dtype = dict()
        for year in range(2007, 2020):
            _cols = self.codebook[self.codebook["year"]
                                  == year]["var_name"].values
            _matched_cols = list()
            for _c in _cols:
                if self._match(_c) in self.valid_columns:
                    self.col2id[_c] = self._match(_c)
                    self.col2dtype[_c] = self.codebook[self.codebook["var_name"]
                                                       == _c]["type_var"].values[0]
                    self.col2dtype[self._match(
                        _c)] = self.codebook[self.codebook["var_name"] == _c]["type_var"].values[0]
                    _matched_cols.append(_c)
            self.year2col[year] = _matched_cols

    def _match(self, x):
        """
        Returns standardized name of the column if possible: XXNNN"""
        pattern = re.compile(r'^([a-zA-Z]{2}).*([0-9]{3})$')
        m = pattern.match(x)
        if m:
            return ("%s%s" % (m.group(1), m.group(2)))
        return None

    def return_valid_column_names(self):
        return list(self.col2id.keys())

    def return_valid_column_transformed_names(self):
        return list(set([i for i in self.col2id.values()]))

    def get_dtype(self, cols):
        result = dict()
        for _c in cols:
            try:
                col_type = self.col2dtype[_c]
            except:
                col_type = None
            if col_type == "categorical":
                result[_c] = tf.categorical
            elif col_type == "numerical":
                result[_c] = tf.numerical
            else:
                result[_c] = tf.numerical

        try:
            del result["data_split"]
        except:
            pass
        return result

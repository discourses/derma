import pandas as pd


class Missing:

    def __init__(self, text=None, numeric=None):
        self.text = 'unknown' if text is None else text
        self.numeric = 0 if numeric is None else numeric

    def homogeneous_replace(self, data: pd.DataFrame, text_fields: list, numeric_fields: list):
        """
        Replaces all NaN values in all\n
            text fields, with the same text string.\n
            numeric fields, with the same number.\n
        :param data: The data frame that hosts the fields listed in text_fields & numeric_fields.\n
        :param text_fields: The list of text fields whose missing values should be replaced.\n
        :param numeric_fields: The list of numeric fields whose missing values should be replaced.\n
        :return data: Updated version of input x
        """

        [data[i].fillna(value=self.text, inplace=True) for i in text_fields]
        [data[i].fillna(value=self.numeric, inplace=True) for i in numeric_fields]

        return data

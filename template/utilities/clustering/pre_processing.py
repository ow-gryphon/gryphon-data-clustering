import pandas as pd
from sklearn.preprocessing import StandardScaler

from .clustering_utilities import generate_column_name


def standardize(frame, variables, std_type, suffix=None):
    """
    Standardize numerical variables in dataset

    :param frame: pandas dataframe
    :param variables: list of numerical variables to standardize
    :param std_type: type of standardization: 'standardize', 'scale', 'demean', '[0,1]','[-1,1]
    :param suffix: suffix for new variables. If None, will use '_std','_scale', '_center', '_unit', '_unit2'

    :return: pandas dataframe with new columns
    """

    new_columns = frame[variables]
    orig_names = new_columns.columns.values
    proposed_names = []

    if std_type == "standardize":
        new_columns = (new_columns - new_columns.mean()) / new_columns.std()
        if suffix is None:
            proposed_names = ["{}_std".format(name) for name in new_columns.columns.values]
        else:
            proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]
    elif std_type == "scale":
        new_columns = new_columns / new_columns.std()
        if suffix is None:
            proposed_names = ["{}_scale".format(name) for name in new_columns.columns.values]
        else:
            proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]
    elif std_type == "demean":
        new_columns = new_columns - new_columns.mean()
        if suffix is None:
            proposed_names = ["{}_center".format(name) for name in new_columns.columns.values]
        else:
            proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]
    elif std_type == "[0,1]":
        new_columns = (new_columns - new_columns.min()) / (new_columns.max() - new_columns.min())
        if suffix is None:
            proposed_names = ["{}_unit".format(name) for name in new_columns.columns.values]
        else:
            proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]
    elif std_type == "[-1,1]":
        new_columns = 2*(new_columns - new_columns.min()) / (new_columns.max() - new_columns.min()) - 1
        if suffix is None:
            proposed_names = ["{}_unit2".format(name) for name in new_columns.columns.values]
        else:
            proposed_names = ["{}{}".format(name, suffix) for name in new_columns.columns.values]

    new_names = [generate_column_name(name, frame) for name in proposed_names]
    new_columns = new_columns.rename(columns=dict(zip(orig_names, new_names)))
    new_frame = pd.concat([frame, new_columns], axis='columns')

    return new_frame


def standardize_variables(data: pd.DataFrame):
    scale = StandardScaler().fit(data)
    no_na_dataset = data.dropna()
    return pd.DataFrame(
        data=scale.transform(no_na_dataset),
        index=no_na_dataset.index,
        columns=data.columns
    )
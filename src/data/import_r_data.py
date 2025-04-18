import pandas as pd  # type: ignore

import rpy2.robjects as robjects  # type: ignore
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr  # type: ignore


def import_r_data(dataset_name: str, package_name: str) -> pd.DataFrame:
    """Import R open dataset from specified package

    Args:
        dataset_name (str): dataset name
        package_name (str): package name with data to import

    Returns:
        pd.DataFrame: imported data
    """
    # activate auto convert
    pandas2ri.activate()
    # import package
    importr(f'{package_name}')
    # load {dataset_name} from {package_name}
    data = robjects.r(f'{dataset_name}')
    # R object -> pd.DataFrame
    return pandas2ri.rpy2py(data)

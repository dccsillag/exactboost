import datetime
import os
import requests
from tqdm import tqdm


def download_from_url(file_url, file_path):
    """
    Downloads files from given url. If file has been partially downloaded it will
    continue the download. Requests are sent in chunks of 1024 bytes. Taken from
    https://gist.github.com/wy193777/0e2a4932e81afc6aa4c8f7a2984f34e2.

    Args:
        file_url (str): string with url to download from.
        file_path (str): file path to save downloaded file to.

    Returns:
        int: file size.
    """
    file_size = int(requests.head(file_url).headers["Content-Length"])
    if os.path.exists(file_path):
        first_byte = os.path.getsize(file_path)
    else:
        first_byte = 0

    retries = 0
    while first_byte < file_size:
        if retries > 0:
            print(f"Current number of retries: {retries}")
        retries += 1
        header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
        pbar = tqdm(
            total=file_size, initial=first_byte,
            unit='B', unit_scale=True, desc=file_url.split('/')[-1])
        req = requests.get(file_url, headers=header, stream=True)
        with(open(file_path, 'ab')) as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(1024)
        pbar.close()
        first_byte = os.path.getsize(file_path)
    print(f"-saved: {file_path}")
    return file_size


def simple_download_from_url(file_url, file_path):
    """
    Alternative function to download files from url, without support for partial
    downloads and chunking. Useful for some websites that block such functionalities
    or file formats that do not allow partial snapshots.

    Args:
        file_url (str): string with url to download from.
        file_path (str): file path to save downloaded file to.
    """
    print(file_url)
    r = requests.get(file_url, allow_redirects=True)
    open(file_path, 'wb').write(r.content)


def get_information_by_date(date, df, frequency="daily"):
    """Returns information of a dataframe in a given date.
    Args:
        date (datetime): The date for retrieving information from df.
        df (pd.DataFrame): DataFrame containing information indexed by date.
        frequency (str): Time frequency of the date.

    Returns:
        pandas.core.series.Series: Row  of df with index date.
    """

    if frequency == "monthly":
        truncated_date = datetime.datetime(date.year, date.month, 1)
        return df.loc[truncated_date]

    if frequency == "daily":
        if date.weekday() < 5:
            return df.loc[date]
        elif date.weekday() == 5:
            return df.loc[
                date - datetime.timedelta(days=1)
            ]  # Deal with saturdays and sundays
        else:
            return df.loc[date - datetime.timedelta(days=2)]


def drop_missing(data, missing_fraction=0.5):
    """Drop columns in DataFrame with more than missing_fraction% of
    missing observations.

    Args:
        data (pd.DataFrame): dataset whose columns are to be dropped if they exhibit
            too much missingness.
        missing_fraction (float): fraction of observations tolerated for each column;
            for example, if 0.5, then columns with more than half of the observations
            missing the feature value will be dropped.

    Returns:
        pd.DataFrame: original dataset with subsetted columns.
    """
    n_obs = data.shape[0]
    missing_fraction_in_col = (data.isnull().sum()/n_obs).sort_values(ascending=False)
    cols_with_excessive_missing = missing_fraction_in_col[
        missing_fraction_in_col > missing_fraction]
    return data.drop(cols_with_excessive_missing.index, axis=1)

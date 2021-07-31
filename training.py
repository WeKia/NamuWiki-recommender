import pandas as pd
import os
import numpy as np

from io import BytesIO
from urllib.request import urlopen
from model.ease_rec import EASE
from zipfile import ZipFile


def download_dataset(dataset, files, data_dir):
    """ Downloads dataset if files are not present. """

    if not np.all([os.path.isfile(data_dir + f) for f in files]):
        url = "http://files.grouplens.org/datasets/movielens/" + dataset.replace('_', '-') + '.zip'
        request = urlopen(url)

        print('Downloading %s dataset' % dataset)

        if dataset in ['ml_100k', 'ml_1m']:
            target_dir = 'raw_data/' + dataset.replace('_', '-')
        elif dataset == 'ml_10m':
            target_dir = 'raw_data/' + 'ml-10M100K'
        else:
            raise ValueError('Invalid dataset option %s' % dataset)

        with ZipFile(BytesIO(request.read())) as zip_ref:
            zip_ref.extractall('raw_data/')

        os.rename(target_dir, data_dir)
        #shutil.rmtree(target_dir)

def main():
    files = ['/u1.base', '/u1.test', '/u.item', '/u.user']
    download_dataset('ml_100k', files, 'data/ml_100k' )

    filename_train = 'data/ml_100k/u1.base'

    sep = '\t'

    dtypes = {
        'u_nodes': np.int32, 'v_nodes': np.int32,
        'ratings': np.float32, 'timestamp': np.float64}

    data_train = pd.read_csv(
        filename_train, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

    print(data_train.values)

    pass

if __name__ == '__main__':
    main()
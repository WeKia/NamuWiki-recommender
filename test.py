import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.models.callbacks import CallbackAny2Vec
from transformers import ElectraTokenizer
from tqdm import tqdm
from functools import partial
import argparse
import pandas as pd
import time
import multiprocessing as mp


parser = argparse.ArgumentParser()
parser.add_argument('--num_process', type=int, help='Number of process that processing graph parse', default=-1)
parser.add_argument('--csv_path', type=str, help='Path to documents information csv file', default='./data/info_processed.csv')
parser.add_argument('--model_path', type=str, help='Path to doc2vec model', default='./data/doc2vec.model')
parser.add_argument('--skip_train', action='store_true', default=False, help='Skip training')
parser.add_argument('--feature', type=str, help='Embedding feature for doc2vec (text, tag). Default is Text', default='text')

args = parser.parse_args()

class MonitorCallback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.last= time.time()

    def on_epoch_end(self, model):
        print('epoch ends {}: {}'.format(self.epoch, time.time() - self.last))
        self.epoch += 1
        self.last = time.time()

def TrainModel(csv_path, num_process, output_path, feature):
    """
    Train Model
    :param csv_path: path for docs info csv file
    :param num_process: number of process for convert docs to TaggedDocument
    :param output_path: output path for model save
    :param feature: feature that model uses
    """
    doc_info = pd.read_csv(csv_path)
    doc_info['category'] = doc_info.category.apply(eval)

    if num_process == -1:
        proc_num = mp.cpu_count() - 1
    else:
        proc_num = num_process

    if proc_num > 1:
        pool = mp.Pool(processes=proc_num)

        data_split = []

        for idx in np.array_split(np.arange(0, len(doc_info)), proc_num):
            data_split.append(doc_info.iloc[idx])

        func = partial(_GetTaggedDoc, feature=feature)

        res = pool.map(func, data_split)
        pool.close()
        pool.join()

        tagged_docs = []

        for r in res:
            tagged_docs += r
    else:
        tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')

        tagged_docs = []

        for index, row in tqdm(doc_info.iterrows(), total=len(doc_info)):
            if feature == 'text':
                tokens = tokenizer.tokenize(row['text'])
            elif feature == 'tag':
                tokens = row['category']
            else:
                raise "Feature not defined"

            tagged_docs.append(TaggedDocument(tokens, [row['title']]))

    print("Data processed!")

    model = Doc2Vec(tagged_docs, vector_size=128, min_count=0, workers=4, epochs=50, callbacks=[MonitorCallback()])

    model.save(output_path)

    return model

def _GetTaggedDoc(data, feature):
    """
    Process tagged Document using tokenizer
    :param data: Pandas dataframe
    :param feature: feature that model uses
    """
    tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')

    tagged_docs = []

    for index, row in tqdm(data.iterrows()):
        if feature == 'text':
            tokens = tokenizer.tokenize(row['text'])
        elif feature == 'tag':
            tokens = row['category']
        else:
            raise "Feature not defined"

        tagged_docs.append(TaggedDocument(tokens, [row['title']]))

    return tagged_docs

if __name__ == '__main__':

    if not args.skip_train:
        model = TrainModel(args.csv_path, args.num_process, args.model_path, args.feature)
    else:
        model = Doc2Vec.load(args.model_path)

    print(model.dv.most_similar('대한민국'))
    print(model.dv.most_similar('서울국제고등학교'))
    print(model.dv.most_similar('드래곤볼'))

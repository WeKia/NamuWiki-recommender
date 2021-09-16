import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from transformers import ElectraTokenizer
from tqdm import tqdm
import argparse
import pandas as pd
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('--num_process', type=int, help='Number of process that processing graph parse', default=-1)
parser.add_argument('--csv_path', type=str, help='Path to documents information csv file', default='./data/info_processed.csv')

args = parser.parse_args()

def _GetTaggedDoc(data):
    """
    Process tagged Document using tokenizer
    :param data: Pandas dataframe
    """
    tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')

    tagged_docs = []

    for index, row in tqdm(data.iterrows()):
        tokens = tokenizer.tokenize(row['text'])

        tagged_docs.append(TaggedDocument(tokens, [row['title']]))

    return tagged_docs

if __name__ == '__main__':
    doc_info = pd.read_csv(args.csv_path)
    doc_info['category'] = doc_info.category.apply(eval)

    if args.num_process == -1:
        proc_num = mp.cpu_count() - 1
    else:
        proc_num = args.num_process

    if proc_num > 1:
        pool = mp.Pool(processes=proc_num)

        data_split = []

        for idx in np.array_split(np.arange(0, len(doc_info)), proc_num):
            data_split.append(doc_info.iloc[idx])

        res = pool.map(_GetTaggedDoc, data_split)
        pool.close()
        pool.join()

        tagged_docs = []

        for r in res:
            tagged_docs += r
    else:
        print("process 1")
        tagged_docs = []

        tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')

        tagged_docs = []

        for index, row in tqdm(doc_info.iterrows(), total=len(doc_info)):
            tokens = tokenizer.tokenize(row['text'])

            tagged_docs.append(TaggedDocument(tokens, [row['title']]))

    print("Data processed!")

    model = Doc2Vec(tagged_docs, vector_size=128, min_count=0, workers=4, epochs=50)

    print(model.dv.most_similar('대한민국'))
    print(model.dv.most_similar('드래곤볼'))

    model.save('./data/doc2vec.model')

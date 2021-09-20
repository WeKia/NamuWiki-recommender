

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
parser.add_argument('--num_process', type=int, help='Number of process that processing graph parse', default=1)
parser.add_argument('--csv_path', type=str, help='Path to documents information csv file', default='./data/info_processed.csv')
parser.add_argument('--model_path', type=str, help='Path to doc2vec model', default='./data/doc2vec.model')
parser.add_argument('--skip_train', action='store_true', default=False, help='Skip training')
parser.add_argument('--vector_size', type=int, help='Size of vector size', default=128)
parser.add_argument('--feature', type=str, help='Embedding feature for doc2vec (text, tag). Default is Text', default='text')

args = parser.parse_args()

class MonitorCallback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.last= time.time()

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('epoch ends {}: {} loss {}'.format(self.epoch, time.time() - self.last, loss))
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
                tags = [row['title']]
            elif feature == 'tag':
                tokens = tokenizer.tokenize(row['text'])
                tags = row['category']
            else:
                raise "Feature not defined"

            tagged_docs.append(TaggedDocument(tokens, tags))

    print("Data processed!")

    model = Doc2Vec(tagged_docs, vector_size=args.vector_size, window=8, dm=1, min_count=0, workers=8, compute_loss=True, epochs=10, callbacks=[MonitorCallback()])

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

def Testing_model(doc_model, tag_vectors, target):
    print('--------------------------------')
    print(f'Target Key : {target}')
    print(doc_model.dv.most_similar(target))

    target_vec = tag_vectors[doc_model.dv.get_index(target)]

    dists = np.dot(tag_vectors, target_vec) / np.linalg.norm(tag_vectors, axis=1)

    topk = np.argsort(dists)[::-1][1:11]

    print([(doc_model.dv.index_to_key[k], dists[k]) for k in topk])

    # Since two vectors has same indices no problem with concatenating
    doc_model.dv.fill_norms()

    norm_tag_vec = tag_vectors / np.linalg.norm(tag_vectors, axis=1)[..., np.newaxis]
    norm_doc_vec = doc_model.dv.vectors / doc_model.dv.norms[..., np.newaxis]

    concat_vector = np.concatenate([norm_tag_vec, norm_doc_vec], axis=1)

    target_vec = concat_vector[doc_model.dv.get_index(target)]

    dists = np.dot(concat_vector, target_vec) / np.linalg.norm(concat_vector, axis=1)

    topk = np.argsort(dists)[::-1][1:11]

    print([(doc_model.dv.index_to_key[k], dists[k]) for k in topk])

if __name__ == '__main__':

    if not args.skip_train:
        model = TrainModel(args.csv_path, args.num_process, args.model_path, args.feature)
    else:
        model = Doc2Vec.load(args.model_path)

    doc_info = pd.read_csv(args.csv_path)
    doc_info['category'] = doc_info.category.apply(eval)

    doc_model = Doc2Vec.load('../data/doc2vec.model')
    tag_model = Doc2Vec.load('../data/doc2vec_tag.model')

    tag_vectors = []

    for i, row in doc_info.iterrows():
        mean = []

        for cat in row['category']:
            mean.append(tag_model[cat])

        if len(mean) == 0:
            mean = [np.ones(64, )]

        mean = np.array(mean).mean(axis=0)
        mean = mean / np.linalg.norm(mean)

        tag_vectors.append(mean)

    tag_vectors = np.array(tag_vectors)

    Testing_model(doc_model, tag_vectors, '대한민국')
    Testing_model(doc_model, tag_vectors, '드래곤볼')
    Testing_model(doc_model, tag_vectors, '과로사(인터넷 방송인)')
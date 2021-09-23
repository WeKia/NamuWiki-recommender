import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.models.callbacks import CallbackAny2Vec
from transformers import ElectraTokenizer
from tqdm import tqdm
from functools import partial
import argparse
import pandas as pd
import time
import json
import tensorflow as tf
import math
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

class NormLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(NormLayer, self).__init__()

    def call(self, input):
        return input / tf.expand_dims(tf.norm(input, axis=1), axis=1)

class MeanPool(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MeanPool, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
          mask = tf.cast(mask, x[0].dtype)
          mask = tf.expand_dims(mask, 2 )
          x *= mask
          return tf.reduce_sum(
              x, axis=1) / tf.reduce_sum(
              mask, axis=1)
        else:
          return tf.reduce_mean(x, axis=1)

    def compute_output_shape(self, input_shape):
          # remove temporal dimension
          return (input_shape[0], input_shape[2])

def get_output_shape_for(self, input_shape):
    # remove temporal dimension
    return input_shape[0], input_shape[2]

def CreateModel(item_length, doc_embeds, tag_embeds):
    item_sequence_input = tf.keras.Input(shape=(20,), name="item_seq_input", dtype='int32')
    doc_embed_layer = tf.keras.layers.Embedding(item_length, doc_embeds.shape[1], weights=[doc_embeds], mask_zero=True,
                                                trainable=False)
    doc_embeddings = doc_embed_layer(item_sequence_input)
    doc_mean_embed = NormLayer()(MeanPool()(doc_embeddings))

    tag_embed_layer = tf.keras.layers.Embedding(item_length, tag_embeds.shape[1], weights=[tag_embeds], mask_zero=True,
                                                trainable=False)
    tag_embeddings = tag_embed_layer(item_sequence_input)
    tag_mean_embed = NormLayer()(MeanPool()(tag_embeddings))

    target_item_input = tf.keras.Input(shape=(1,), name="target_item")
    target_doc_embed = doc_embed_layer(target_item_input)
    target_doc_embed = NormLayer()(MeanPool()(target_doc_embed))
    target_tag_embed = tag_embed_layer(target_item_input)
    target_tag_embed = NormLayer()(MeanPool()(target_tag_embed))

    item_concat = tf.keras.layers.Concatenate(axis=1)([doc_mean_embed, tag_mean_embed])

    target_concat = tf.keras.layers.Concatenate(axis=1)([target_doc_embed, target_tag_embed])

    output = tf.keras.layers.Dot(axes=1)([item_concat, target_concat])

    model = tf.keras.Model(inputs=[item_sequence_input, target_item_input], outputs=output)

    return model

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

    dummy = np.array([[1., 1., 1.],
                      [2., 2., 2.]])

    #print( MeanPool()(dummy, mask=np.array([[True, True, False], [True, True, True]])))


    doc_info = pd.read_csv(args.csv_path)
    doc_info['category'] = doc_info.category.apply(eval)

    doc_model = Doc2Vec.load('./model/pretrain/doc2vec.model')
    tag_model = Doc2Vec.load('./model/pretrain/doc2vec_tag.model')

    doc_vector = np.array(doc_model.dv.vectors)

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

    # Add Pad index
    doc_vector = np.concatenate([np.zeros(doc_vector.shape[1]).reshape(1, -1), doc_vector])

    tag_vectors = np.array(tag_vectors)
    # Add Pad index
    tag_vectors = np.concatenate([np.zeros(tag_vectors.shape[1]).reshape(1, -1), tag_vectors])

    model = CreateModel(tag_vectors.shape[0], doc_vector, tag_vectors)

    model.summary()

    model.save('testmodel.h5')

    batch_size = 256
    batched_input = np.tile(np.array([[doc_model.dv.get_index('드래곤볼') + 1] + [0 for _ in range(19)]]), (batch_size, 1))

    print(batched_input.shape)

    Scores = [];

    loop_length = math.ceil(379440 / batch_size);

    for i in tqdm(range(loop_length)):
        item_batch = None
        if (i < loop_length - 1):
            item_batch = np.arange(batch_size * i, batch_size * (i + 1));
        else:
            batched_input = np.tile(np.array([[doc_model.dv.get_index('드래곤볼') + 1] + [0 for _ in range(19)]]), ((379440 % batch_size), 1))
            item_batch = np.arange(batch_size * i, batch_size * i + (379440 % batch_size));

        output = model.predict([batched_input, item_batch]).tolist();

        Scores += output

    topk = np.argsort(np.array(Scores).reshape(-1))[::-1][1:11]

    print([(doc_model.dv.index_to_key[k - 1]) for k in topk])
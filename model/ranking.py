import argparse
import numpy as np
import pandas as pd
import time
import math
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from utils.evaluation import f1score, precision, recall
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, help='Path to documents information csv file', default=None)
parser.add_argument('--tagmodel_path', type=str, help='Pretrained tag doc2vec model path.', default='./pretrain/doc2vec_tag.model')
parser.add_argument('--docmodel_path', type=str, help='Pretrained text base doc2vec model path.', default='./pretrain/doc2vec.model')
parser.add_argument('--train_path', type=str, help='Path to train data csv file', default='../data/train.csv')
parser.add_argument('--valid_path', type=str, help='Path to valid data(path to valid_te, valid_tr)', default='../data/valid')

args = parser.parse_args()

def CreateModel(item_length, doc_embeds, tag_embeds):
    """
    :param item_length: length of total items
    :param doc_embeds : pretrained embeddings for doc2vec
    :param tag_embeds : pretrained embeddings for documents tags
    """

    item_sequence_input = tf.keras.Input(shape=(100,), name="item_seq_input", dtype='int32')
    doc_embed_layer = tf.keras.layers.Embedding(item_length, doc_embeds.shape[1], weights=[doc_embeds], mask_zero=True, trainable=False)
    doc_embeddings = doc_embed_layer(item_sequence_input)
    doc_mean_embed = tf.keras.layers.GlobalAveragePooling1D()(doc_embeddings)

    tag_embed_layer = tf.keras.layers.Embedding(item_length, tag_embeds.shape[1], weights=[tag_embeds], mask_zero=True, trainable=False)
    tag_embeddings = tag_embed_layer(item_sequence_input)
    tag_mean_embed = tf.keras.layers.GlobalAveragePooling1D()(tag_embeddings)

    target_item_input = tf.keras.Input(shape=(1,), name="target_item")
    target_doc_embed = doc_embed_layer(target_item_input)
    target_tag_embed = tag_embed_layer(target_item_input)

    concat = tf.keras.layers.Concatenate(axis=1)([doc_mean_embed, tag_mean_embed, target_doc_embed, target_tag_embed])
    x = tf.keras.layers.Dropout(0.5)(concat)
    x = tf.keras.layers.Dense(512, name="dense1")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1)(x)  # (batch, embed_size)
    output = tf.keras.layers.Activation('sigmoid')(output)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=1000, decay_rate=0.97)

    model = tf.keras.Model(inputs=[item_sequence_input, target_item_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),  # Optimizer
        # Loss function to minimize
        loss=tf.keras.losses.BinaryCrossentropy(),
        # List of metrics to monitor
        metrics=[precision, recall, f1score],
    )

    return model

def GetCandidate(Base_vectors, target_vector, topk=100):
    """
    Get Candidate set from Base vectors with target vector
    :param Base_vectors:
    :param target_vector:
    :param topk: How many Candidates need
    """
    dists = tf.keras.layers.Dot(axes=(2, 2))([Base_vectors[np.newaxis, ...], target_vector[np.newaxis, ...]])
    dists = tf.transpose(tf.squeeze(dists))

    topk = tf.argsort(dists, axis=1).numpy()[:, -(topk + 1):-1]
    return topk

class SimpleBatchData(Sequence):
    def __init__(self, data, batch_size=128):
        """
        Simple data for batching
        :param data:
        :param batch_size:
        """
        super(SimpleBatchData, self).__init__()
        self.data = data
        self.batch_size = batch_size

    def __getitem__(self, idx):
        data = np.array(self.data[idx*self.batch_size:(idx+1)*self.batch_size])

        return data

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

class EmbedData(Sequence):
    def __init__(self, users, doc_dict, doc_vectors, tag_vectors, batch_size=128, max_len=100, test_data=None):
        """
            Preprocess Data. model takes Embedding of docs
            To more faster training and inference, embed docs before it goes to model.

            :param users: User data contains user's preference
            :type users: Pandas DataFrame

            :param doc_dict: dict of documents id and index
            :type doc_dict: dict(int)

            :param max_len: Maximum length of users contributed documents.
                if num of users contributed documents is less than this val, pad it.
            :type max_len: int

            :param batch_size:

        """
        super(EmbedData, self).__init__()

        self.batch_size = batch_size
        self.item_seq = []
        self.target = []
        self.label = []

        # Since item_seq is too big to store for every target, we just index offset that refers item sequence
        self.offset = []

        doc_means, tag_means = [], []

        for i, (user, docs) in tqdm(enumerate(users.items()), total=len(users)):
            pos_docs = [doc_dict[doc] for doc in docs]
            if len(docs) > max_len:
                seq = np.random.choice(pos_docs, max_len, replace=False)
            else:
                seq = pos_docs + [0 for _ in range(max_len - len(docs))]

            self.item_seq.append(seq)

            # Init mean vectors for doc, tag models
            doc_mean, tag_mean = [], []

            for doc in pos_docs:
                doc_mean.append(doc_vectors[doc])
                tag_mean.append(tag_vectors[doc])

            doc_mean = np.array(doc_mean).mean(axis=0)
            doc_mean = doc_mean / np.linalg.norm(doc_mean)  # Normalize

            tag_mean = np.array(tag_mean).mean(axis=0)
            tag_mean = tag_mean / np.linalg.norm(tag_mean)  # Normalize

            doc_means.append(doc_mean)
            tag_means.append(tag_mean)

        doc_means = np.array(doc_means)
        tag_means = np.array(tag_means)

        batched_Docmeans = SimpleBatchData(doc_means, 256)
        batched_Tagmeans = SimpleBatchData(tag_means, 256)

        candidates = []

        for doc_mean, tag_mean in tqdm(zip(batched_Docmeans, batched_Tagmeans), total=len(batched_Tagmeans)):
            doc_cands = GetCandidate(doc_vectors, doc_mean, 100)
            tag_cands = GetCandidate(tag_vectors, tag_mean, 100)

            cands = np.concatenate([doc_cands, tag_cands], axis=1)
            candidates += cands.tolist()

        for i, (user, docs) in tqdm(enumerate(users.items()), total=len(users)):
            for cand in candidates[i]:
                if cand in docs:
                    label = 1
                else:
                    label = 0

                self.target.append(cand)
                self.label.append(label)
                self.offset.append(i)

            if test_data is not None:
                try:
                    for doc in test_data[user]:
                        self.target.append(doc_dict[doc])
                        self.label.append(1)
                        self.offset.append(i)
                except:
                    # There are some cases user is in tr but not in te
                    continue

        self.item_seq = np.array(self.item_seq)
        self.target = np.array(self.target)
        self.label = np.array(self.label)
        self.offset = np.array(self.offset)

    def __getitem__(self, idx):
        item_seq = np.array(self.item_seq[self.offset[idx*self.batch_size:(idx+1)*self.batch_size]])
        targets = self.target[idx*self.batch_size:(idx+1)*self.batch_size]
        labels = self.label[idx*self.batch_size:(idx+1)*self.batch_size]

        return (item_seq, targets), labels

    def __len__(self):
        return math.ceil(len(self.target) / self.batch_size)

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

def main():
    doc_info = pd.read_csv(args.csv_path)
    doc_info['category'] = doc_info.category.apply(eval)

    doc_model = Doc2Vec.load(args.docmodel_path)
    tag_model = Doc2Vec.load(args.tagmodel_path)

    doc_vector = np.array(doc_model.dv.vectors)

    # Add Pad index
    doc_vector = np.concatenate([np.zeros(doc_vector.shape[1]).reshape(1, -1), doc_vector])

    tag_vectors = []

    # Order of doc titles and doc2vec key must be same
    for a, b in zip(doc_model.dv.key_to_index.keys(), doc_info.title):
        assert a==b

    for i, row in tqdm(doc_info.iterrows(), total=len(doc_info)):
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
    tag_vectors = np.concatenate([np.zeros(tag_vectors.shape[1]).reshape(1, -1), tag_vectors])

    # Testing_model(doc_model, tag_vectors, '대한민국')
    # Testing_model(doc_model, tag_vectors, '드래곤볼')
    # Testing_model(doc_model, tag_vectors, '과로사(인터넷 방송인)')

    doc_dict = {id: i + 1 for i, id in enumerate(list(doc_info.id))}
    item_length = tag_vectors.shape[0]

    train_raw = pd.read_csv(args.train_path)
    train_raw = train_raw.groupby('contributors')['title'].apply(list)
    valid_te_raw = pd.read_csv(args.valid_path + '_te.csv')
    valid_te_raw = valid_te_raw.groupby('contributors')['title'].apply(list)

    valid_tr_raw = pd.read_csv(args.valid_path + '_tr.csv')
    valid_tr_raw = valid_tr_raw.groupby('contributors')['title'].apply(list)

    print(valid_te_raw)

    valid_data = EmbedData(valid_tr_raw, doc_dict, doc_vector, tag_vectors, batch_size=4096, test_data=valid_te_raw)
    traindata = EmbedData(train_raw, doc_dict, doc_vector, tag_vectors, batch_size=4096)

    model = CreateModel(item_length, doc_vector, tag_vectors)
    model.summary()

    checkpoint_filepath = 'bestmodel.h5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_f1score',
        mode='max',
        save_best_only=True)

    steps_per_epoch = 10000
    epochs = 10 * (len(traindata) // steps_per_epoch + 1)

    model.fit(traindata, validation_data=valid_data, steps_per_epoch=steps_per_epoch,
              callbacks=[model_checkpoint_callback], epochs=epochs)

if __name__ == '__main__':
    main()
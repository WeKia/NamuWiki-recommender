import argparse
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from utils.evaluation import f1score, precision, recall
from tqdm import tqdm

def CreateModel(item_length, embed_size, doc_embeds, tag_embeds):
    """
    :param item_length: length of total items
    :param embed_size: dimension of embedding layer
    :param doc_embeds : pretrained embeddings for doc2vec
    :param tag_embeds : pretrained embeddings for documents tags
    """

    item_sequence_input = tf.keras.Input(shape=(100,), name="item_seq_input", dtype='int32')
    doc_embed_layer = tf.keras.layers.Embedding(item_length, embed_size, weights=[doc_embeds], mask_zero=True, trainable=False)
    doc_embeddings = doc_embed_layer(item_sequence_input)
    doc_mean_embed = tf.keras.layers.GlobalAveragePooling1D()(doc_embeddings)

    tag_embed_layer = tf.keras.layers.Embedding(item_length, embed_size, weights=[tag_embeds], mask_zero=True, trainable=False)
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

class EmbedData(Sequence):
    def __init__(self, users, doc_dict, batch_size=128, max_len=100):
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

        doc_ids = list(doc_dict.values())

        doc_ids.sort()
        max_key = doc_ids[-1]


        for i, (user, docs) in tqdm(enumerate(users.items()), total=len(users)):
            pos_docs = [doc_dict[doc] for doc in docs]
            if len(docs) > max_len:
                seq = np.random.choice(pos_docs, max_len, replace=False)
            else:
                seq = pos_docs + [0 for _ in range(max_len - len(docs))]

            self.item_seq.append(seq)

            for doc in pos_docs:
                self.target.append(doc)
                self.label.append(1)
                self.offset.append(i)

            test_pos_doc = []
            if isTest:
                test_pos_doc = [doc_dict[doc] for doc in test_data[user]]

                for doc in test_pos_doc:
                    self.target.append(doc)
                    self.label.append(1)
                    self.offset.append(i)

            mask[pos_docs + test_pos_doc] = False
            negatives = arr[mask]
            mask[pos_docs + test_pos_doc] = True

            if not isTest:
                # More faster than np.choice
                idx = np.random.randint(len(negatives), size=neg_samplesize)
                negative_samples = [negatives[i] for i in idx]
            else:
                idx = np.random.randint(len(negatives), size=neg_samplesize * 10)
                negative_samples = [negatives[i] for i in idx]

            for doc in negative_samples:
                self.target.append(doc)
                self.label.append(0)
                self.offset.append(i)

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


def main():
    pass

if __name__ == '__main__':
    main()
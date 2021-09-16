import argparse
import numpy as np
import pandas as pd
import json
import math
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import ElectraModel, ElectraTokenizer
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
from utils.evaluation import f1score, precision, recall

parser = argparse.ArgumentParser()
parser.add_argument('--num_process', type=int, help='Number of process that processing graph parse', default=-1)
parser.add_argument('--max_length', type=int, help='Maximum length of category', default=64)
parser.add_argument('--skip_train', action='store_true', default=False, help='Skip training')
parser.add_argument('--csv_path', type=str, help='Path to documents information csv file', default=None)
parser.add_argument('--train_path', type=str, help='Path to train data csv file', default='./data/train.csv')
parser.add_argument('--valid_path', type=str, help='Path to valid data(path to valid_te, valid_tr)', default='./data/valid')
parser.add_argument('--test_path', type=str, help='Path to test data(path to test_te, test_tr)', default='./data/test')
parser.add_argument('--preembed_path', type=str, help='Path to pre embedded data path', default='./data/preembed.npy')
parser.add_argument('--epochs', type=int, help='Epochs for learning', default=5)
parser.add_argument('--lr', type=float, help="Learning rate for model", default=0.0005)
parser.add_argument('--max_grad_norm', type=float, help='max_grad_norm for model', default=0.5)
parser.add_argument('--log_interval', type=int, help='Log interval for model training', default=1000)
parser.add_argument('--device', type=str, help='Device for training model', default='cpu')
parser.add_argument('--model_path', type=str, help='Pretrained model path. If not exits do training.', default='./model/pretrain/model.pt')

args = parser.parse_args()

class EmbedData(Sequence):
    def __init__(self, users, doc_dict, batch_size=128, max_len=100, neg_samplesize=1000, isTest=False, test_data=None):
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

            :param neg_samplesize: negative sample size
            :type iteration: int

            :param isTest: If data is test(valid) data, all negative data must be sampled
            :type isTest: bool

            :param batch_size:

            :param test_data: If it is test data, then get data that is not in users item sequence
                            but user contributed
            :type test_data : Pandas DataFrame
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

        # For doing fast negative sampling, make masks and array
        arr = np.array([n for n in range(max_key + 1)])
        mask = np.zeros_like(arr)
        mask[doc_ids] = 1

        # Make mask to bool type
        mask = mask > 0

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

def CreateModel(item_length, embed_size, side_embeds):
    """

    :param item_length: length of total items
    :param embed_size: dimension of embedding layer
    """

    item_sequence_input = tf.keras.Input(shape=(100,), name="item_seq_input", dtype='int32')
    item_embed_layer = tf.keras.layers.Embedding(item_length, embed_size, mask_zero=True)
    item_embed = item_embed_layer(item_sequence_input)
    mean_embed = tf.keras.layers.GlobalAveragePooling1D()(item_embed)

    side_embed = tf.keras.layers.Embedding(item_length, 256, weights=[side_embeds], mask_zero=True, trainable=False)(item_sequence_input)
    side_mean = tf.keras.layers.GlobalAveragePooling1D()(side_embed)

    concat = tf.keras.layers.Concatenate(axis=1)([mean_embed, side_mean])
    x = tf.keras.layers.Dropout(0.5)(concat)
    x = tf.keras.layers.Dense(512, name="dense1")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    user_embed = tf.keras.layers.Dense(embed_size)(x)  # (batch, embed_size)

    target_item_input = tf.keras.Input(shape=(1,), name="target_item")
    target_embed = item_embed_layer(target_item_input)  # (batch, embed_size)

    dot = tf.keras.layers.Dot(axes=[1, 2])([user_embed, target_embed])
    output = tf.keras.layers.Activation('sigmoid')(dot)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=1000, decay_rate=0.97)

    model = tf.keras.Model(inputs=[item_sequence_input, target_item_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),  # Optimizer
        # Loss function to minimize
        loss=tf.keras.losses.BinaryCrossentropy(),
        # List of metrics to monitor
        metrics=[precision, recall, f1score],
    )

    return model

def Preembed_Docs(doc_info, device='cuda'):
    """
    Preprocessing document texts and categories.
    Embed them using KoElectra

    :param doc_info: Document information dataframe
    :type doc_info: Pandas DataFrame
    """

    text = doc_info['text'].to_numpy()

    tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
    electra = ElectraModel.from_pretrained("monologg/koelectra-small-v3-discriminator")
    electra.to(device)

    textLoader = DataLoader(text, batch_size=16)

    # idx 0 is Pad embed ( all 0)
    embedding = np.array([[0 for _ in range(256)]])
    for batch in tqdm(textLoader):
        token = tokenizer(batch, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        token['input_ids'] = token['input_ids'].to(device)
        token['attention_mask'] = token['attention_mask'].to(device)
        embed = electra(input_ids=token['input_ids'], attention_mask=token['attention_mask'])[0][:, 0]

        embedding = np.concatenate([embedding, embed.detach().cpu().numpy()])


    with open(args.preembed_path, 'wb') as f:
        np.save(f, embedding)

    return embedding

if __name__ == '__main__':
    doc_info = pd.read_csv(args.csv_path)
    doc_info['category'] = doc_info.category.apply(eval)

    text_dict = None
    if (args.preembed_path is None) or not os.path.exists(args.preembed_path):
        print("No preembeded data found!")
        text_embed = Preembed_Docs(doc_info)
    else:
        text_embed = np.load(args.preembed_path)

    doc_dict = {id: i + 1 for i, id in enumerate(list(doc_info.id))}
    item_length = text_embed.shape[0]

    train_raw = pd.read_csv(args.train_path)
    train_raw = train_raw.groupby('contributors')['title'].apply(list)

    valid_te_raw = pd.read_csv(args.valid_path + '_te.csv')
    valid_te_raw = valid_te_raw.groupby('contributors')['title'].apply(list)

    valid_tr_raw = pd.read_csv(args.valid_path + '_tr.csv')
    valid_tr_raw = valid_tr_raw.groupby('contributors')['title'].apply(list)

    valid_data = EmbedData(valid_tr_raw, doc_dict, batch_size=4096, isTest=True, test_data=valid_te_raw)
    traindata = EmbedData(train_raw, doc_dict, batch_size=4096, neg_samplesize=2000)

    model = CreateModel(item_length, 256, text_embed)
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
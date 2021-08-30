import argparse
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import time
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import ElectraModel, ElectraTokenizer
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--num_process', type=int, help='Number of process that processing graph parse', default=-1)
parser.add_argument('--max_length', type=int, help='Maximum length of category', default=64)
parser.add_argument('--skip_train', action='store_true', default=False, help='Skip training')
parser.add_argument('--csv_path', type=str, help='Path to documents information csv file', default=None)
parser.add_argument('--train_path', type=str, help='Path to train data csv file', default='../data/train.csv')
parser.add_argument('--valid_path', type=str, help='Path to valid data(path to valid_te, valid_tr)', default='../data/valid')
parser.add_argument('--test_path', type=str, help='Path to test data(path to test_te, test_tr)', default='../data/test')
parser.add_argument('--preembed_path', type=str, help='Path to pre embedded data path', default='../data/preembed.npy')
parser.add_argument('--epochs', type=int, help='Epochs for learning', default=5)
parser.add_argument('--lr', type=float, help="Learning rate for model", default=0.0005)
parser.add_argument('--max_grad_norm', type=float, help='max_grad_norm for model', default=0.5)
parser.add_argument('--log_interval', type=int, help='Log interval for model training', default=1000)
parser.add_argument('--device', type=str, help='Device for training model', default='cpu')
parser.add_argument('--model_path', type=str, help='Pretrained model path. If not exits do training.', default='../model/pretrain/model.pt')

args = parser.parse_args()

class YoutubeCandidate(nn.Module):
    def __init__(self, input_size, hidden_layers, ouput_size=256):
        """
            Youtube Candidate model

            :param input_size: User data contains user's preference
            :type input_size: int

            :param hidden_layers: PreEmbedded documents texts
            :type hidden_layers: List(int)

            :param ouput_size: User Embed size
            :type ouput_size: int
        """
        super(YoutubeCandidate, self).__init__()

        layers = []

        for layer_size in hidden_layers:
            layers += [nn.Linear(input_size, layer_size),
                       nn.BatchNorm1d(layer_size),
                       nn.ReLU(inplace=True),
                       nn.Dropout(p=0.5)]

            input_size = layer_size

        layers.append(nn.Linear(input_size, ouput_size))

        self.hidden = nn.Sequential(*layers)

    def user_embed(self, item_seq):
        """
        Embed item sequences to use embed

        :param item_seq: Mean ebmed vector from Users contributed items(docs)
        :type item_seq: tensor, (batch_size, squence_embedsize)
        """
        x = self.hidden(item_seq)

        return x

    def forward(self, item_seq, target_embed):
        """
        Compute logits of target embed and user embed given by item sequences
        reutnr dot product of user embed and target_embed

        :param item_seq: Mean ebmed vector from Users contributed items(docs)
        :type item_seq: tensor, (batch_size, squence_embedsize)

        :param target_embed : Target item embeddings
        :type target_embed : Tensor, (batch_size, squence_embedsize)
        """
        user_embed = self.user_embed(item_seq)

        return (user_embed * target_embed).sum(dim=1)

class EmbedData(Dataset):
    def __init__(self, users, text_dict, category_dict=None, max_len=100, neg_samplesize=1000):
        """
            Preprocess Data. model takes Embedding of docs
            To more faster training and inference, embed docs before it goes to model.

            :param users: User data contains user's preference
            :type users: Pandas DataFrame

            :param text_dict: PreEmbedded documents texts
            :type text_dict: Dict(array)

            :param category_dict: PreEmbedded documents categories
            :type category_dict: Dict(array)

            :param max_len: Maximum length of users contributed documents.
                if num of users contributed documents is less than this val, pad it.
            :type max_len: int

            :param neg_samplesize: negative sample size
            :type iteration: int
        """
        super(EmbedData, self).__init__()

        self.embeds = []
        self.target = []
        self.label = []

        doc_ids = list(text_dict.keys())
        doc_ids.sort()
        max_key = doc_ids[-1]

        # For doing fast negative sampling, make masks and array
        arr = np.array([n for n in range(max_key + 1)])
        mask = np.zeros_like(arr)
        mask[doc_ids] = 1

        # Make mask to bool type
        mask = mask > 0

        for user, docs in tqdm(users.items(), total=len(users)):
            text_embed = []
            for doc in docs:
                text_embed.append(torch.tensor(text_dict[doc]))

            embed = torch.mean(torch.stack(text_embed), dim=0)

            for doc in docs:
                self.embeds.append(embed)
                self.target.append(text_dict[doc])
                self.label.append(1)

            mask[docs] = False
            negatives = arr[mask]
            mask[docs] = True

            # More faster than np.choice
            idx = np.random.randint(len(negatives), size=neg_samplesize)
            negative_samples = [negatives[i] for i in idx]

            for doc in negative_samples:
                self.embeds.append(embed)
                self.target.append(text_dict[doc])
                self.label.append(0)

    def __getitem__(self, idx):
        return self.embeds[idx], self.target[idx], self.label[idx]

    def __len__(self):
        return len(self.embeds)

class TestingData:
    def __init__(self, users, text_dict, target_dict, max_target=20, max_hist=100):
        """
            Preprocess Data. model takes Embedding of docs
            This object is different with simple embedData
            EmbedData returns embedding, target, label
            But this object returns embeddings, targets(list), history
            history is list of users previous view documents

            :param users: User data contains user's preference
            :type users: Pandas DataFrame

            :param text_dict: PreEmbedded documents texts
            :type text_dict: Dict(array)

            :param target_dict : Diction maps user to target documents
            :type target_dict : Dict(int, int)

            :param max_target: Pad number to maximum length of targets
            :type max_target: int

            :param max_hist : Pad number to maximum length of history
            :type max_hist : int
        """
        super(TestingData, self).__init__()

        #indexing document id to numpy idx
        idx_dict = {doc: i for i, doc in enumerate(text_dict.keys())}

        self.embeds = []
        self.target = []
        self.history = []

        for user, docs in tqdm(users.items(), total=len(users)):
            text_embed = []
            for doc in docs:
                text_embed.append(torch.tensor(text_dict[doc]))

            embed = torch.mean(torch.stack(text_embed), dim=0)

            self.embeds.append(embed)
            assert len(docs) <= max_hist
            assert len(target_dict[user]) <= max_target
            self.history.append([idx_dict[doc] for doc in docs ] + [-1 for _ in range(max_hist - len(docs))])
            self.target.append([idx_dict[doc] for doc in target_dict[user]] + [-1 for _ in range(max_target - len(target_dict[user]))])

    def __getitem__(self, idx):
        return self.embeds[idx], np.array(self.target[idx]), np.array(self.history[idx])

    def __len__(self):
        return len(self.embeds)

def _tokenize(data, id, batch_size=512, max_len=64):
    """
        Tokenize data

        :param data: Strings to tokenize
        :type data: list(str)

        :param id: process id
        :type id: int

        :param batch_size: batch size of tokenizer and KoElectra
        :type batch_size: int

        :param max_len: Max length of category, text length is already processed
        :type max_len: int
    """
    tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
    electra = ElectraModel.from_pretrained("monologg/koelectra-small-v3-discriminator")
    dataLoader = DataLoader(data, batch_size=batch_size)

    embedding = None
    for batch in dataLoader:
        with torch.no_grad():
            token = tokenizer(batch, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')
            embed = electra(input_ids=token['input_ids'], attention_mask=token['attention_mask'])[0][:, 0]

        if embedding is not None:
            embedding = np.concatenate([embedding, embed])
        else:
            embedding = embed

    return embedding, id

def Preembed_Docs(doc_info, device='cuda'):
    """
    Preprocessing document texts and categories.
    Embed them using KoElectra

    :param doc_info: Document information dataframe
    :type doc_info: Pandas DataFrame
    """

    if args.num_process == -1:
        proc_num = mp.cpu_count() - 1
    else:
        proc_num = args.num_process

    text = doc_info['text'].to_numpy()
    func = partial(_tokenize, max_len=512)

    print("Process text tokenize")
    s = time.time()
    # text_embeds = pool.starmap(func, zip(np.array_split(text, proc_num), range(proc_num)))
    # pool.close()
    # pool.join()
    #
    # text_embeds.sort(key=lambda x: x[1])
    # text_embed = np.concatenate([embed[0] for embed in text_embeds])
    print(f"Process done after {time.time() - s}")

    tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
    electra = ElectraModel.from_pretrained("monologg/koelectra-small-v3-discriminator")
    electra.to(device)

    textLoader = DataLoader(text, batch_size=256)

    embedding = None
    for batch in tqdm(textLoader):
        with torch.no_grad():
            token = tokenizer(batch, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
            token['input_ids'] = token['input_ids'].to(device)
            token['attention_mask'] = token['attention_mask'].to(device)
            embed = electra(input_ids=token['input_ids'], attention_mask=token['attention_mask'])[0][:, 0]

        if embedding is not None:
            embedding = np.concatenate([embedding, embed.cpu().numpy()])
        else:
            embedding = embed.cpu().numpy()

    with open(args.preembed_path, 'wb') as f:
        np.save(f, embedding)

    text_dict = dict(zip(doc_info.id, embedding))
    return text_dict

def TrainModel(model, dataloader, optimizer, loss_fn, epochs=5,
               log_interval=1000, device='cpu', max_grad_norm=1):
    """
    Train Model with data

    :param model: Model to train
    :param dataloader: DataLoader generated from dataset
    :param optimizer: Optimzier for model
    :param loss_fn: Loss function
    :param epochs:
    :param log_interval:
    :param device:
    :param max_grad_norm:
    """

    for epoch in range(epochs):
        model.train()
        i = 0
        for item_seq, target, label in tqdm(dataloader):
            item_seq = item_seq.to(device)
            target = target.to(device)
            label = label.float().to(device)

            logits = model(item_seq, target)

            optimizer.zero_grad()

            loss = loss_fn(logits, label)
            loss.backward()

            sig = torch.sigmoid(logits)
            pred = torch.zeros_like(label)
            pred[sig >= 0.5] = 1

            f1score = f1_score(label.cpu(), pred.cpu())

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            i += 1
            if (i % log_interval) == 0:
                print(f"Epoch {epoch} train loss {loss.data.cpu()}")
                print(f"Epoch {epoch} f1 score {f1score}")
        
def TestModel(model, TestDataLoader, doc_embeds,device='cuda'):
    """
    Testing model by computing recommedendation

    :param model:
    :param TestDataLoader:
    :param doc_embeds: document embeddings for all documents
    By multiply user embeds and doc_embeds get documnets score
    """
    for embed, target, history in tqdm(TestDataLoader):
        embed = embed.to(device)

        cont_docs = torch.cat([target, history], dim=1).long().cpu()

        batch_size = embed.size(0)
        labels = torch.zeros((batch_size, doc_embeds.size(0))).cpu()

        label_list = []

        for label, docs in zip(labels.unbind(), cont_docs.unbind()):
            label[torch.masked_select(docs, docs != -1)] = 1
            label_list.append(label.cpu())

        labels = torch.vstack(label_list).long()

        user_embed = model.user_embed(embed)

        item_scores = torch.matmul(user_embed, doc_embeds.T).cpu()

        sig = torch.sigmoid(item_scores)
        pred = torch.zeros_like(labels)
        pred[sig >= 0.5] = 1

        f1 = 0

        for true, hat in zip(labels.unbind(), pred.unbind()):
            f1 += f1_score(true, hat)

        print(f1 / batch_size)

        top_items = torch.argsort(item_scores, dim=1)[:, :10].cpu().numpy()
        target = target.cpu().numpy()

        topk_acc = 0

        for item, t in zip(top_items, target):
            print(item)
            intersect = np.intersect1d(item, t)

            topk_acc += len(intersect) / 10

        print(topk_acc / batch_size)

    pass

def main():

    doc_info = pd.read_csv(args.csv_path)
    doc_info['category'] = doc_info.category.apply(eval)

    text_dict = None
    if (args.preembed_path is None) or not os.path.exists(args.preembed_path):
        print("No preembeded data found!")
        text_dict = Preembed_Docs(doc_info)
    else:
        text_embed = np.load(args.preembed_path)
        text_dict = dict(zip(doc_info.id, text_embed))

    model = YoutubeCandidate(256, [1024, 512])
    model.to(args.device)

    if not args.skip_train:
        train_raw = pd.read_csv(args.train_path)
        train_raw = train_raw.groupby('contributors')['title'].apply(list)

        train_data = EmbedData(train_raw, text_dict)

        dataloader = DataLoader(train_data, batch_size=512, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        TrainModel(model, dataloader, optimizer, loss_fn, args.epochs,
                   args.log_interval, args.device, args.max_grad_norm)

        torch.save(model.state_dict(), args.model_path)

    model.load_state_dict(torch.load(args.model_path))

    valid_te_raw = pd.read_csv(args.valid_path + '_te.csv')
    valid_te_raw = valid_te_raw.groupby('contributors')['title'].apply(list)

    valid_tr_raw = pd.read_csv(args.valid_path + '_tr.csv')
    valid_tr_raw = valid_tr_raw.groupby('contributors')['title'].apply(list)

    target_dict = valid_te_raw.to_dict()

    validData = TestingData(valid_tr_raw, text_dict, target_dict)
    validLoader = DataLoader(validData, batch_size=256)

    doc_embeds = torch.tensor(list(text_dict.values())).to(args.device)

    TestModel(model, validLoader, doc_embeds)

    pass

if __name__ == '__main__':
    main()

    # tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
    #
    # print(tokenizer(test[0]))
    #
    # testloader = DataLoader(test, batch_size=4)
    #
    # for i in testloader:
    #     print(tokenizer(i))
import pandas as pd
import argparse
import numpy as np
import ast
import time
import datetime as dt
import multiprocessing as mp
import sys
import os
from functools import partial

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path for csv file')
    parser.add_argument('--contributor_path', type=str, default='contributor.csv',
                        help='Path for contributor file')
    parser.add_argument('--info_path', type=str, default='info.csv',
                        help='Path for document info file')
    parser.add_argument('--proc_num', type=int, help='Number of parsing processor default is -1 that is max core of cpu', default=-1)
    parser.add_argument('--use_category', action='store_true', default=False,
                        help='Use category instead of title')
    parser.add_argument('--convert_idx', action='store_true', default=False, help='Convert item, users to idx')
    parser.add_argument('--doc_min', type=int, help='Restrict minimum number of how many documents contributors contributed', default=20)
    parser.add_argument('--doc_max', type=int,help='Restrict maximum number of how many documents contributors contributed', default=100)
    parser.add_argument('--cont_min', type=int, help='Restrict minimum number of contributors of document ', default=5)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--divide_tag', action='store_true', default=False, help='Divide tag(category) if there is / on tag')

    args = parser.parse_args()

    return args

# Since some documents are redirected to other document, move contributor of them to redirected document
def process_contributors(titles, contributors, relink):
    out = {}

    for title, cont in zip(titles, contributors):
        link = relink.get(title)
        if link:
            # Since there are secondary redirected documents, explore non-redirect document
            while relink.get(link) is not None:
                if link == relink.get(link):
                    break
                link = relink.get(link)

            if out.get(link):
                out[link].extend(cont)
            else:
                out[link] = cont

        else:
            if out.get(title):
                out[title].extend(cont)
            else:
                out[title] = cont

    return out

# Multiprocessing data, These codes are from Advances in Financial Machine Learning by Marcos Lopez de Prado
def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    out=[]
    for job in jobs:
        out_=expandCall(job)
        out.append(out_)
    return out

def linParts(numAtoms,numThreads):
    parts=np.linspace(0,numAtoms,min(numThreads,numAtoms)+1)
    parts=np.ceil(parts).astype(int)
    return parts

def reportProgress(jobNum,numJobs,time0,task):
    # Report progress as asynch jobs are completed
    msg=[float(jobNum)/numJobs,(time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp=str(dt.datetime.fromtimestamp(time.time()))
    msg=timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ \
    str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
    if jobNum<numJobs:sys.stderr.write(msg+'\r')
    else:sys.stderr.write(msg+'\n')
    return

def expandCall(kargs):
    # Expand the arguments of a callback function, kargs[’func’]
    func=kargs['func']
    del kargs['func']
    out=func(**kargs)
    return out

def processJobs(jobs,task=None,numThreads=24):
    # Run in parallel.
    # jobs must contain a ’func’ callback, for expandCall
    if task is None:task=jobs[0]['func'].__name__
    pool=mp.Pool(processes=numThreads)
    outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time()
    # Process asynchronous output, report progress
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join() # this is needed to prevent memory leaks
    return out

def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,**kargs):
    parts = linParts(len(pdObj), numThreads * mpBatches)

    jobs = []
    for i in range(1, len(parts)):
        obj = pdObj.iloc[parts[i - 1]:parts[i]]
        job = {'titles': obj['title'], 'contributors' : obj['contributors'], 'func': func}
        job.update(kargs)
        jobs.append(job)

    print(parts)

    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads=numThreads)

    print(out)
    diction = {}

    for o in out:
        for k, v in o.items():
            if diction.get(k) :
                diction[k] = diction[k].extend(v)
            else:
                diction[k] = v

    print(diction)

    return diction

def make_csv(args):

    csv = pd.read_csv(args.csv_path, keep_default_na=False, na_values=[''])

    if args.proc_num == -1:
        process_num = mp.cpu_count() - 1
    else:
        process_num = args.proc_num

    # Since pandas writes list as string, retrieve them
    csv['links'] = csv.links.apply(lambda x: ast.literal_eval(x)[0])
    csv['contributors'] = csv.contributors.apply(lambda x: ast.literal_eval(x)[0])
    csv['category'] = csv.category.apply(lambda x: ast.literal_eval(x)[0])
    csv['link_length'] = csv.links.apply(lambda x: len(x))
    redirect = csv[(csv.link_length == 1) & (csv.text.str.contains('redirect'))]
    csv = csv[~((csv.link_length == 1) & (csv.text.str.contains('redirect')))]

    relink = {}

    for index, row in redirect.iterrows():
        relink[row['title']] = row['links'][0]

    diction = {}
    category_transformer = {}
    i = 0

    for index, row in csv.iterrows():
        link = relink.get(row['title'])
        if link:
            # Since there are secondary redirected documents, explore non-redirect document
            while relink.get(link) is not None:
                if link == relink.get(link):
                    break
                link = relink.get(link)

            if diction.get(link):
                diction[link].extend(row['contributors'])
            else:
                diction[link] = row['contributors']

            if category_transformer.get(link):
                category_transformer[link].extend(row['category'])
            else:
                category_transformer[link] = row['category']

        else:
            if diction.get(row['title']):
                diction[row['title']].extend(row['contributors'])
            else:
                diction[row['title']] = row['contributors']

            if category_transformer.get(row['title']):
                category_transformer[row['title']].extend(row['category'])
            else:
                category_transformer[row['title']] = row['category']

        i += 1

        if (i % 10000 == 0):
            print(f"processed {i}")

    print('process done')

    # processed = mpPandasObj(process_contributors, csv, numThreads=process_num, relink=relink)
    # processed = pd.DataFrame(list(processed.items()), columns=['title', 'contributors'])
    processed = pd.DataFrame(list(diction.items()), columns=['title', 'contributors'])
    if args.use_category:
        processed['title'] = processed.title.apply(lambda x : category_transformer[x])

    processed = processed.explode('contributors')
    processed = processed.explode('title')

    processed = processed.drop_duplicates().dropna()

    # contributor_count = processed.groupby('title').count()
    # restricted_docs = contributor_count[contributor_count['contributors'] >= args.cont_min].index.tolist()
    # processed = processed[processed['title'].isin(restricted_docs)]
    #
    # doc_count = processed.groupby('contributors').count()
    # restricted_conts = doc_count[(doc_count['title'] >= args.doc_min) & (doc_count['title'] <= args.doc_max)].index.tolist()
    # processed = processed[processed['contributors'].isin(restricted_conts)]
    #
    # print(processed.groupby('contributors').count().describe())
    # print(processed.groupby('title').count().describe())
    #
    # csv = csv[csv['title'].isin(pd.unique(processed['title']))]

    if args.convert_idx:
        unique_title = pd.unique(processed['title'])
        doc2id = dict((did, i) for (i, did) in enumerate(unique_title))
        cont2id = dict((cont, i) for (i, cont) in enumerate(pd.unique(processed['contributors'])))

        processed['title'] = processed.title.apply(lambda x : doc2id[x])
        processed['contributors'] = processed.contributors.apply(lambda x : cont2id[x])
        csv['id'] = csv.title.apply(lambda x : doc2id[x])
        csv['links'] = csv.links.apply(lambda x : [doc2id[link] for link in x if doc2id.get(link)])

    if args.divide_tag:
        def divide_tag(x):
            tags = []
            for e in x:
                tags += e.split('/')


            return list(set(tags))

        csv['category'] = csv.category.apply(lambda x : divide_tag(x))


    processed.to_csv(args.contributor_path, index=False, encoding='utf-8-sig')

    csv = csv.drop('contributors', axis=1)
    csv.to_csv(args.info_path, index=False, encoding='utf-8-sig')

    with open(os.path.join('unique_sid.txt'), 'w', -1, 'utf-8') as f:
        for did in pd.unique(processed['title']):
            f.write('%s\n' % did)


def main(args):
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    np.random.seed(2)

    if (not os.path.exists(args.contributor_path)) or (not os.path.exists(args.info_path)):
        print("No preprocessed csv found")
        make_csv(args)

    contributors = pd.read_csv(args.contributor_path)
    docs = pd.read_csv(args.info_path)

    # restrict mimum and maximum
    contributor_count = contributors.groupby('title').count()
    restricted_docs = contributor_count[contributor_count['contributors'] >= args.cont_min].index.tolist()
    contributors = contributors[contributors['title'].isin(restricted_docs)]

    doc_count = contributors.groupby('contributors').count()
    restricted_conts = doc_count[(doc_count['title'] >= args.doc_min) & (doc_count['title'] <= args.doc_max)].index.tolist()
    contributors = contributors[contributors['contributors'].isin(restricted_conts)]

    n_cont = len(restricted_conts)
    val_size = int(n_cont * args.val_ratio)
    test_size = int(n_cont * args.test_ratio)

    np.random.shuffle(restricted_conts)

    train_user = restricted_conts[:(n_cont - val_size - test_size)]
    val_user = restricted_conts[(n_cont - val_size - test_size) : (n_cont - test_size)]
    test_user = restricted_conts[(n_cont - test_size):]

    # Convert titles and contributors name to integer id
    # May be converting title is not used
    doc2id = dict((did, i) for (i, did) in enumerate(restricted_docs))
    cont2id = dict((cont, i) for (i, cont) in enumerate(restricted_conts))

    train_data = contributors[contributors['contributors'].isin(train_user)]

    # Since we train on training data, remove documents that not appear in training data
    unique_did = pd.unique(train_data['title'])
    doc2id = dict((did, i) for (i, did) in enumerate(unique_did))

    # Code from Enhancing VAEs for Collaborative Filtering: Flexible Priors & Gating Mechanisms
    # https://github.com/psywaves/EVCF

    def split_train_test_proportion(data, test_prop=0.2):
        data_grouped_by_user = data.groupby('contributors')
        tr_list, te_list = list(), list()

        np.random.seed(2)

        for i, (_, group) in enumerate(data_grouped_by_user):
            n_items_u = len(group)

            if n_items_u >= 5:
                idx = np.zeros(n_items_u, dtype='bool')
                idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

                tr_list.append(group[np.logical_not(idx)])
                te_list.append(group[idx])
            else:
                tr_list.append(group)

            if i % 1000 == 0:
                print("%d contributors sampled" % i)
                sys.stdout.flush()

        data_tr = pd.concat(tr_list)
        data_te = pd.concat(te_list)

        return data_tr, data_te

    valid_data = contributors[contributors['contributors'].isin(val_user)]
    valid_data = valid_data[valid_data['title'].isin(unique_did)]

    test_data = contributors[contributors['contributors'].isin(test_user)]
    test_data = test_data[test_data['title'].isin(unique_did)]

    valid_train_data, valid_test_data = split_train_test_proportion(valid_data)
    test_train_data, test_test_data = split_train_test_proportion(test_data)

    docs = docs[docs['id'].isin(unique_did)]

    train_data.to_csv('train.csv', index=False, encoding='utf-8-sig')
    valid_train_data.to_csv('valid_tr.csv', index=False, encoding='utf-8-sig')
    valid_test_data.to_csv('valid_te.csv', index=False, encoding='utf-8-sig')
    test_train_data.to_csv('test_tr.csv', index=False, encoding='utf-8-sig')
    test_test_data.to_csv('test_te.csv', index=False, encoding='utf-8-sig')

    docs.to_csv('info_processed', index=False, encoding='utf-8-sig')

    # with open(os.path.join('unique_sid.txt'), 'w', -1, 'utf-8') as f:
    #     for did in unique_did:
    #         f.write('%s\n' % did)


if __name__ == '__main__':
    args = get_parser()
    main(args)
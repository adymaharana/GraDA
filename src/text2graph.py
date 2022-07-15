import json, csv
import spacy
from collections import defaultdict
from tqdm import tqdm
import os, pickle
import editdistance
import numpy as np
import pprint
from copy import deepcopy
import random

nlp = spacy.load('en_core_web_sm')


def load_conceptnet(conceptnet_file, relations=None):
    # kb = defaultdict(lambda: defaultdict(lambda: []))
    kb = {}
    with open(conceptnet_file, 'r', encoding='utf-8') as f:
        kb_all = f.readlines()
    print("Reading Concept Net file and building KB")
    for line in tqdm(kb_all):
        e1, relation, e2 = [token.strip() for token in line.split(',')]

        if '_' in e1:
            continue
        if '_' in e2:
            continue

        if editdistance.eval(e1, e2) <= 5:
            continue

        if relations is not None and relation not in relations:
            continue
        # kb[relation][e1].append(e2)
        # kb[relation + '-inv'][e2].append(e1)
        if e1 not in kb:
            kb[e1] = {}
        if relation not in kb[e1]:
            kb[e1][relation] = []
        kb[e1][relation].append(e2)
        if e2 not in kb:
            kb[e2] = {}
        inv_relation = relation + '-inv'
        if inv_relation not in kb[e2]:
            kb[e2][inv_relation] = []
        kb[e2][inv_relation].append(e1)

    # for rel in kb.keys():
    #    print(rel)

    return kb


def load_conceptnet(conceptnet_file, relations=None):
    #kb = defaultdict(lambda: defaultdict(lambda: []))
    kb = {}
    with open(conceptnet_file, 'r', encoding='utf-8') as f:
        kb_all = f.readlines()
    print("Reading Concept Net file and building KB")
    for line in tqdm(kb_all):
        e1, relation, e2 = [token.strip() for token in line.split(',')]

        if '_' in e1:
            continue
        if '_' in e2:
            continue

        if editdistance.eval(e1, e2) <= 5:
            continue
        
        if relations is not None and relation not in relations:
            continue
        #kb[relation][e1].append(e2)
        #kb[relation + '-inv'][e2].append(e1)
        if e1 not in kb:
            kb[e1] = {}
        if relation not in kb[e1]:
            kb[e1][relation] = []
        kb[e1][relation].append(e2)
        if e2 not in kb:
            kb[e2] = {}
        inv_relation = relation + '-inv'
        if inv_relation not in kb[e2]:
            kb[e2][inv_relation] = []
        kb[e2][inv_relation].append(e1)

    #for rel in kb.keys():
    #    print(rel)

    return kb


def filter_cnet(cnet):
    relations = ['AtLocation', 'Causes', 'CapableOf', 'Antonym', 'HasSubevent', 'HasPrerequisite',
                 'CausesDesire', 'Desires', 'PartOf', 'HasProperty']
    relations.extend([rel + '-inv' for rel in relations])

    #for relation in cnet.keys():
    #    if relation not in relations:
    #        del cnet[relation]


    subjects = list(cnet.keys())
    for subject in subjects:
        if len(subject.split('_')) > 1:
            del cnet[subject]
            continue
        for relation in cnet[subject].keys():
            objects = cnet[subject][relation]
            for obj in objects:
                if len(obj.split('_')) > 1:
                    cnet[subject][relation].remove(obj)
                    continue
                if editdistance.eval(subject, obj) <= 5:
                    cnet[subject][relation].remove(obj)

    subjects = list(cnet.keys())
    for subject in subjects:
        if cnet[subject] is None or cnet[subject] == {}:
            del cnet[subject]
            continue
        relation_keys = list(cnet[subject].keys())
        for relation in relation_keys:
            if cnet[subject][relation] is None or cnet[subject][relation] == []:
                del cnet[subject][relation]

    print("Filtered ConceptNet")
    return cnet


cnet = load_conceptnet('../data/cn_relations_orig.txt')
#print(list(cnet.keys()))
cnet = filter_cnet(cnet)


def find_content_words(spacy_doc):
    content_pos_tags = ['VERB', 'NOUN']
    content_tokens = []
    for token in spacy_doc:
        if token.pos_ in content_pos_tags:
            content_tokens.append(token.text.lower())
    return content_tokens

def read_jsonl(data_file):
    with open(data_file, encoding="utf-8") as f:
        samples = [json.loads(line.strip()) for line in f.readlines()]
    return samples

def read_txt(data_file):
    with open(data_file, encoding="utf-8") as f:
        data = [line.strip() for line in f.readlines()]
    return data

def get_cnet_nodes(cnet, words, max_hops=1):

    word2node = {}
    subjs = list(cnet.keys())

    for word in words:
        edit_dists = [editdistance.eval(sub, word) for sub in subjs]
        min_idx = np.argmin(edit_dists)
        if edit_dists[min_idx] >= 5:
            word2node[word] = None
        else:
            word2node[word] = subjs[min_idx]

    found = sum([1 if word2node[word] is not None else 0 for word in words])
    print('%s / %s words found in ConceptNet' % (found, len(word2node)))
    pprint.pprint(word2node)

def get_multi_hop_path(cnet, word1, word2):

    paths = []
    objs1 = []
    obj_counts1 = []
    relations1 = list(cnet[word1].keys())
    
    for rel in relations1:
        objs1.extend(cnet[word1][rel])
        obj_counts1.append(len(cnet[word1][rel]))

    objs2 = []
    obj_counts2 = []
    relations2 = list(cnet[word2].keys())
    
    for rel in relations2:
        objs2.extend(cnet[word2][rel])
        obj_counts2.append(len(cnet[word2][rel]))

    common_words = list(set(objs1) & set(objs2))
    if len(common_words) == 0:
        return None
    else:
        for word in common_words:
            idx1 = objs1.index(word)
            idx2 = objs2.index(word)

            rel1 = [rel for i, rel in enumerate(relations1) if sum(obj_counts1[:max(0, i+1)]) >= idx1][0]
            rel2 = [rel for i, rel in enumerate(relations2) if sum(obj_counts2[:max(0, i+1)]) >= idx2][0]
            paths.append((word1, rel1, word, rel2, word2))

            try:
                assert (word in cnet[word1][rel1]) and (word2 in cnet[word][rel2]), 'wut??'
            except:
                continue
    return paths

def get_cnet_path(cnet, words, max_hops=1):

    word2node = {}
    subjs = list(cnet.keys())
    paths = []
    for i in range(len(words)):
        word1 = words[i]
        if word1 not in subjs:
            continue
        
        for j, word2 in enumerate(words[i+1:]):
            if word2 not in subjs:
                continue

            for rel in cnet[word1].keys():
                if word2 in cnet[word1][rel]:
                    paths.append((word1, rel, word2))

            multi_hop_paths = get_multi_hop_path(cnet, word1, word2)
            if multi_hop_paths:
                paths.extend(multi_hop_paths)

    return paths

def read_csv(data_file):
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for row in csv_reader:
            data.append(row)

    return data

def get_cnet_paths_codah(data_file, out_file):

    samples = read_csv(data_file)
    with open(out_file, 'w') as f:
        for sample in tqdm(samples):
            sentence = sample[1]
            doc = nlp(sentence)
            content_words = find_content_words(doc)
            paths = get_cnet_path(cnet, content_words)
            f.write(json.dumps(paths) + '\n')

def get_cnet_paths_hellaswag(data_file, out_file):

    samples = read_jsonl(data_file)
    with open(out_file, 'w') as f:
        for sample in tqdm(samples):
            sentence = sample["ctx"]
            doc = nlp(sentence)
            content_words = find_content_words(doc)
            paths = get_cnet_path(cnet, content_words)
            f.write(json.dumps(paths) + '\n')

def prepare_gpt_train_files_codah():

    samples = read_csv('../data/codah/full_data.tsv')

    paths = []
    with open('../data/codah/full_data_cnet_paths.jsonl', 'r') as f:
        for line in f.readlines():
            paths.append(json.loads(line.strip()))

    idxs = list(range(len(samples)))
    train_idxs = random.sample(idxs, k=int(len(idxs)*0.9))
    val_idxs = [i for i in range(len(idxs)) if i not in train_idxs]
    print(max(val_idxs), max(train_idxs))
    print('%s and %s samples in train and validation' % (len(train_idxs), len(val_idxs)))

    fquestion = open('../data/codah/train_gpt_codah_question.jsonl', 'w')
    fanswer = open('../data/codah/train_gpt_codah_answer.jsonl', 'w')
    foption = open('../data/codah/train_gpt_codah_option.jsonl', 'w')\

    fquestion_val = open('../data/codah/valid_gpt_codah_question.jsonl', 'w')
    fanswer_val = open('../data/codah/valid_gpt_codah_answer.jsonl', 'w')
    foption_val = open('../data/codah/valid_gpt_codah_option.jsonl', 'w')

    for i, sample in enumerate(samples):

        label = int(sample[-1])
        options = sample[2:-1]
        assert len(options) == 4
        answer = options[label]
        options.remove(answer)

        if i in val_idxs:
            fanswer_val.write(json.dumps({'input': sample[1], 'target': answer}) + '\n')
            for option in options:
                foption_val.write(json.dumps({'input': sample[1], 'target': option}) + '\n')
        else:
            fanswer.write(json.dumps({'input': sample[1], 'target': answer}) + '\n')
            for option in options:
                foption.write(json.dumps({'input': sample[1], 'target': option}) + '\n')
            continue

    path_idxs = [i for i, path in enumerate(paths) if path != []]
    path_train_idxs = random.sample(path_idxs, k=int(len(path_idxs)*0.9))
    path_val_idxs = [i for i in path_idxs if i not in path_train_idxs]
    print('%s and %s samples in train and validation using paths' % (len(path_train_idxs), len(path_val_idxs)))

    for i, (sample, path) in enumerate(zip(samples, paths)):

        if i not in path_idxs:
            continue

        content_words = []
        for p in path:
            content_words.append(p[0])
            content_words.append(p[-1])
        content_words = list(set(content_words))
        random.shuffle(content_words)

        if i in path_val_idxs:
            fquestion_val.write(json.dumps({'content_words': content_words, 'target': sample[1]}) + '\n')
        else:
            fquestion.write(json.dumps({'content_words': content_words, 'target': sample[1]}) + '\n')

    fquestion.close()
    fanswer.close()
    foption.close()
    fquestion_val.close()
    fanswer_val.close()
    foption_val.close()

def get_generation_paths_winogrande(data_file, k=50000):

    samples = read_jsonl(data_file)
    new_samples = []
    for _ in range(k):
        candidate_words = [s['content_words'] for s in random.choices(samples, k=5)]
        random.shuffle(candidate_words)
        new_samples.append(deepcopy(candidate_words[:int(len(candidate_words)/5)]))

    with open('data/winogrande-train-generate-words.jsonl', 'w') as f:
        for sample in new_samples:
            f.write(json.dumps(sample) + '\n')

def get_generation_paths(cnet, path_file, out_file, n_mutations=1):

    paths = []
    with open(path_file, 'r') as f:
        for line in f.readlines():
            paths.append(json.loads(line.strip()))

    new_samples = []
    skipped = 0
    for _ in range(n_mutations):
        for i, path in enumerate(paths):
            if len(path) == 0:
                continue
            elif len(path) <= 5:
                pass
            else:
                path = random.sample(path, k=5)

            raw_path = random.choice(path)
            subj = raw_path[-3]
            rel = raw_path[-2]

            try:
                new_obj = random.choice(cnet[subj][rel])
            except:

                if '-inv' in rel:
                    rel = rel[:-4]
                else:
                    rel = rel + '-inv'
                try:
                    new_obj = random.choice(cnet[subj][rel])
                except:
                    print(skipped, raw_path)
                    print(cnet[subj].keys())
                    skipped += 1
                    continue


                # continue

            new_path = raw_path[:-1] + [new_obj]
            path.remove(raw_path)
            path.append(new_path)

            content_words = []
            for p in path:
                content_words.append(p[0])
                content_words.append(p[-1])
            content_words = list(set(content_words))
            random.shuffle(content_words)
            new_samples.append({'content_words': content_words})

    with open(out_file, 'w') as f:
        for sample in new_samples:
            f.write(json.dumps(sample) + '\n')

def prepare_gpt_train_files_hellaswag():

    samples = read_jsonl('../data/hellaswag/hellaswag_2k_train.jsonl')

    paths = []
    with open('../data/hellaswag/hellaswag_2k_train_cnet_paths.jsonl', 'r') as f:
        for line in f.readlines():
            paths.append(json.loads(line.strip()))

    fquestion = open('../data/hellaswag/hellaswag_2k_train_gpt_question.jsonl', 'w')
    fanswer = open('../data/hellaswag/hellaswag_2k_train_gpt_answer.jsonl', 'w')
    foption = open('../data/hellaswag/hellaswag_2k_train_gpt_options.jsonl', 'w')

    for i, sample in enumerate(samples):

        label = int(sample["label"])
        options = sample["endings"]
        assert len(options) == 4
        answer = options[label]
        options.remove(answer)

        fanswer.write(json.dumps({'input': sample["ctx"], 'target': answer}) + '\n')
        for option in options:
            foption.write(json.dumps({'input': sample["ctx"], 'target': option}) + '\n')
        continue

    for i, (sample, path) in enumerate(zip(samples, paths)):

        if path == []:
            continue

        content_words = []
        for p in path:
            content_words.append(p[0])
            content_words.append(p[-1])
        content_words = list(set(content_words))
        random.shuffle(content_words)

        fquestion.write(json.dumps({'content_words': content_words, 'target': sample["ctx"]}) + '\n')

    fquestion.close()
    fanswer.close()
    foption.close()


if __name__ == "__main__":

    #get_cnet_paths_winogrande('../data/winogrande/train_xl.jsonl')
    #get_cnet_paths_winogrande('../data/siqa/train_merged-fold-0.jsonl')
    #get_generation_paths_random_walk(cnet, '../data/winogrande/train_xl-words.jsonl')
    #get_generation_paths_winogrande('../data/winogrande/train_xl-words.jsonl', k=75000)

    # get_cnet_paths_codah('../data/codah/full_data.tsv', '../data/codah/full_data_cnet_paths.jsonl')
    # prepare_gpt_train_files_codah()
    # get_generation_paths(cnet, '', '', 2)

    # get_cnet_paths_hellaswag('../data/hellaswag/hellaswag_2k_train.jsonl', '../data/hellaswag/hellaswag_2k_train_cnet_paths.jsonl')
    # prepare_gpt_train_files_hellaswag()
    get_generation_paths(cnet, '../data/hellaswag/hellaswag_2k_train_cnet_paths.jsonl', '../data/hellaswag/hellaswag_generation_paths_10k.jsonl', 5)
import csv
import sys
import os
import torch
from torch.utils.data import TensorDataset
csv.field_size_limit(sys.maxsize)
import numpy as np
from tokenizers import SentencePieceBPETokenizer
import pickle

class AGBDataReader(object):
    """
    Reads in the Heidelberg AGB dataset
    """
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.tokenizer = SentencePieceBPETokenizer("./data/sentencepiece_tokenizer/vocab.json",
                                                   "./data/sentencepiece_tokenizer/merges.txt")

    def get_examples(self, filename,max_seq_length=1024, max_examples=0,read_cache=False):
        """
        data_splits specified which data split to use atrain, dev, test).
        Expects that self.dataset_folder contains the files in tsv (tab-separated form),
        with three columns (s1 \t s2 \t [0|1]
        """

        #load from saved features to save time
        if read_cache:
            name=filename.replace(".tsv","")
            if max_examples>0:
                name+"_"+str(max_examples)
            with open('./data/dataset_'+name+".pickle", 'rb') as file:
                dataset = pickle.load(file)
        else:
            with open(os.path.join(self.dataset_folder, filename)) as f:
                rows = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)

                first_sections = []
                second_sections=[]
                labels=[]
                id = 0
                for sentence_a, sentence_b, label in rows:
                    id += 1
                    sentence_a= self.tokenizer.encode(sentence_a).ids
                    sentence_b=self.tokenizer.encode(sentence_b).ids


                    padding = [0] * (max_seq_length - len(sentence_a))
                    sentence_a += padding
                    padding = [0] * (max_seq_length - len(sentence_b))
                    sentence_b += padding


                    first_sections.append(np.array(sentence_a[:max_seq_length]))
                    second_sections.append(np.array(sentence_b[:max_seq_length]))
                    labels.append(self.map_label(label))

                    if 0 < max_examples <= len(first_sections):
                        break
            print(len(first_sections))
            dataset = TensorDataset(torch.LongTensor(np.stack(first_sections,axis=0)),torch.LongTensor(np.stack(second_sections,axis=0)), torch.FloatTensor(np.array(labels)))
            #save the features
            name = filename.replace(".tsv", "")
            if max_examples > 0:
                name + "_" + str(max_examples)
            with open('./data/dataset_'+name+".pickle", 'wb') as file:
                pickle.dump(dataset, file, protocol=4)
        return dataset

    @staticmethod
    def get_labels():
        # Adding different types of labels to assert correct conversion
        return {"same_section": 1, "other_section": 0, "1": 1, "0": 0, 1: 1, 0: 0}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]





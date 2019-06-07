import os
import json
import numpy as np

from cached_property import cached_property

from typing import Dict, List, Tuple, Set, Optional


class Chinese_selection_preprocessing(object):
    def __init__(self, hyper):
        self.hyper = hyper
        self.raw_data_root = hyper.raw_data_root
        self.data_root = hyper.data_root
        self.schema_path = os.path.join(self.raw_data_root, 'all_50_schemas')

        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(
                'schema file not found, please check your downloaded data!')
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)

        self.relation_vocab_path = os.path.join(self.data_root,
                                                hyper.relation_vocab)

    @cached_property
    def relation_vocab(self):
        if os.path.exists(self.relation_vocab_path):
            pass
        else:
            self.gen_relation_vocab()
        return json.load(open(self.relation_vocab_path, 'r'))

    def gen_relation_vocab(self):
        relation_vocab = {}
        i = 0
        for line in open(self.schema_path, 'r'):
            relation = json.loads(line)['predicate']
            if relation not in relation_vocab:
                relation_vocab[relation] = i
                i += 1
        relation_vocab['N'] = i
        json.dump(relation_vocab,
                  open(self.relation_vocab_path, 'w'),
                  ensure_ascii=False)

    def read_line(self, line: str) -> Optional[str]:
        line = line.strip("\n")
        if not line:
            return None
        instance = json.loads(line)
        text = instance['text']

        bio = None
        selection = None

        if 'spo_list' in instance:
            spo_list = instance['spo_list']

            if not self.check_valid(text, spo_list):
                return None
            spo_list = [{
                'predicate': spo['predicate'],
                'object': spo['object'],
                'subject': spo['subject']
            } for spo in spo_list]

            entities: List[str] = self.spo_to_entities(text, spo_list)
            relations: List[str] = self.spo_to_relations(text, spo_list)

            bio = self.spo_to_bio(text, entities)
            selection = self.spo_to_selection(text, spo_list)

        result = {
            'text': text,
            'spo_list': spo_list,
            'bio': bio,
            'selection': selection
        }
        return json.dumps(result, ensure_ascii=False)

    def gen_one_data(self, dataset):
        source = os.path.join(self.raw_data_root, dataset)
        target = os.path.join(self.data_root, dataset)
        with open(source, 'r') as s, open(target, 'w') as t:
            for line in s:
                newline = self.read_line(line)
                if newline is not None:
                    t.write(newline)
                    t.write('\n')

    def gen_all_data(self):
        self.gen_one_data(self.hyper.train)
        self.gen_one_data(self.hyper.dev)

    def check_valid(self, text: str, spo_list: List[Dict[str, str]]) -> bool:
        if spo_list == []:
            return False
        if len(text) > self.hyper.max_text_len:
            return False
        for t in spo_list:
            if t['object'] not in text or t['subject'] not in text:
                return False
        return True

    def spo_to_entities(self, text: str,
                        spo_list: List[Dict[str, str]]) -> List[str]:
        entities = set(t['object'] for t in spo_list) | set(t['subject']
                                                            for t in spo_list)
        return list(entities)

    def spo_to_relations(self, text: str,
                         spo_list: List[Dict[str, str]]) -> List[str]:
        return [t['predicate'] for t in spo_list]

    def spo_to_selection(self, text: str, spo_list: List[Dict[str, str]]
                         ) -> List[Dict[str, int]]:

        selection = []
        for triplet in spo_list:

            object = triplet['object']
            subject = triplet['subject']

            object_pos = text.find(object) + len(object) - 1
            relation_pos = self.relation_vocab[triplet['predicate']]
            subject_pos = text.find(subject) + len(subject) - 1

            selection.append({
                'subject': subject_pos,
                'predicate': relation_pos,
                'object': object_pos
            })

        return selection

    def spo_to_bio(self, text: str, entities: List[str]) -> List[str]:
        bio = ['O'] * len(text)
        for e in entities:
            begin = text.find(e)
            end = begin + len(e) - 1

            assert end <= len(text)

            bio[begin] = 'B'
            for i in range(begin + 1, end + 1):
                bio[i] = 'I'
        return bio

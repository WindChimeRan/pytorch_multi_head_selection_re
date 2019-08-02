from typing import Dict, List, Tuple, Set, Optional
from abc import ABC, abstractmethod
from overrides import overrides


class F1_abc(object):
    def __init__(self):
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    def reset(self) -> None:
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    def get_metric(self, reset: bool = False):
        if reset:
            self.reset()

        f1, p, r = 2 * self.A / (self.B +
                                 self.C), self.A / self.B, self.A / self.C
        result = {"precision": p, "recall": r, "fscore": f1}

        return result

    def __call__(self, predictions,
                 gold_labels):
        raise NotImplementedError


class F1_triplet(F1_abc):

    @overrides
    def __call__(self, predictions: List[List[Dict[str, str]]],
                 gold_labels: List[List[Dict[str, str]]]):

        for g, p in zip(gold_labels, predictions):
            try:
                g_set = set('_'.join((gg['object'], gg['predicate'],
                                    gg['subject'])) for gg in g)
                p_set = set('_'.join((pp['object'], pp['predicate'],
                                    pp['subject'])) for pp in p)
            except:
                g_set = set('_'.join((''.join(gg['object']), gg['predicate'],
                                    ''.join(gg['subject']))) for gg in g)
                p_set = set('_'.join((''.join(pp['object']), pp['predicate'],
                                    ''.join(pp['subject']))) for pp in p)
            self.A += len(g_set & p_set)
            self.B += len(p_set)
            self.C += len(g_set)


class F1_ner(F1_abc):

    @overrides
    def __call__(self, predictions: List[List[str]], gold_labels: List[List[str]]):
        for g, p in zip(gold_labels, predictions):

            inter = sum(tok_g == tok_p and tok_g in ('B', 'I')
                        for tok_g, tok_p in zip(g, p))
            bi_g = sum(tok_g in ('B', 'I') for tok_g in g)
            bi_p = sum(tok_p in ('B', 'I') for tok_p in p)

            self.A += inter
            self.B += bi_g
            self.C += bi_p

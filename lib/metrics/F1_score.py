from typing import Dict, List, Tuple, Set, Optional
from abc import ABC, abstractmethod
from overrides import overrides

from collections import namedtuple
from collections import defaultdict

class F1_abc(object):
    def __init__(self):
        self.A = 0
        self.B = 0
        self.C = 0
        self.ABC = namedtuple('ABC', ['A', 'B', 'C'])
        self.rel_detail = defaultdict(list)

    def reset(self) -> None:
        self.A = 0
        self.B = 0
        self.C = 0
        self.rel_detail.clear()

    def get_metric(self, reset: bool = False):
        if reset:
            self.reset()
        result = self.calc(self.A, self.B, self.C)
        return result

    @staticmethod
    def calc(A, B, C):
        p = A / B if B > 0 else 0.
        r = A / C if C > 0 else 0.
        f1 = 2 * p * r / (p + r) if (p+r) > 0 else 0.
        prf1 = {"precision": p, "recall": r, "fscore": f1}
        return prf1

    @staticmethod
    def calc_abc(A, B, C):
        p = A / B if B > 0 else 0.
        r = A / C if C > 0 else 0.
        f1 = 2 * p * r / (p + r) if (p+r) > 0 else 0.
        prf1 = {"precision": p, "recall": r, "fscore": f1, "ABC": "{}:{}:{}".format(A, B, C), "sum":A+B+C}
        return prf1

    def get_metric_detail(self, reset: bool = False):
        if reset:
            self.reset()
        results = {}
        for k, v in self.rel_detail.items():
            results[k] = self.calc_abc(v[0], v[1], v[2])
        return results

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

            # for rel detail
            g_set_rel, p_set_rel = defaultdict(list), defaultdict(list)
            try:
                for gg in g:
                    g_set_rel[gg['predicate']].append('_'.join((gg['object'], gg['predicate'], gg['subject']))) 
                for pp in p:
                    p_set_rel[pp['predicate']].append('_'.join((pp['object'], pp['predicate'], pp['subject']))) 
            except:
                for gg in g:
                    g_set_rel[gg['predicate']].append('_'.join((''.join(gg['object']), gg['predicate'], ''.join(gg['subject']))))
                for pp in p:
                    p_set_rel[gg['predicate']].append('_'.join((''.join(pp['object']), pp['predicate'], ''.join(pp['subject']))))

            rels = set(list(g_set_rel.keys()) + list(p_set_rel.keys()))
            for k in rels:
                if k not in self.rel_detail:
                    self.rel_detail[k] = [0, 0, 0]

            for k in rels:
                vg, vp = g_set_rel.get(k, []), p_set_rel.get(k, [])
                self.rel_detail[k][0] += len(set(vg) & set(vp))
                self.rel_detail[k][1] += len(set(vp))
                self.rel_detail[k][2] += len(set(vg))


class F1_ner(F1_abc):

    @overrides
    def __call__(self, predictions: List[List[str]], gold_labels: List[List[str]]):
        for g, p in zip(gold_labels, predictions):

            inter = sum(tok_g == tok_p and tok_g in ('B', 'I')
                        for tok_g, tok_p in zip(g, p))
            bi_g = sum(tok_g in ('B', 'I') for tok_g in g)
            bi_p = sum(tok_p in ('B', 'I') for tok_p in p)

            self.A += inter
            self.B += bi_p
            self.C += bi_g

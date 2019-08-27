import os
import json
import time
from datetime import datetime
import argparse
import codecs
import torch

from typing import Dict, List, Tuple, Set, Optional

from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

from torch.optim import Adam, SGD
from pytorch_transformers import AdamW, WarmupLinearSchedule

from lib.preprocessings.chinese_selection import Chinese_selection_preprocessing
from lib.preprocessings.conll_selection import Conll_selection_preprocessing
from lib.preprocessings.conll_bert_selecetion import Conll_bert_preprocessing
from lib.dataloaders.selection_loader import Selection_Dataset, Selection_loader, Selection_loader_predict
from lib.metrics.F1_score import F1_triplet, F1_ner
from lib.models.selection import MultiHeadSelection
from lib.config.hyper import Hyper
from lib.preprocessings.medical_selection import Medical_selection_preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',
                    '-e',
                    type=str,
                    default='conll_bert_re',
                    help='experiments/exp_name.json')
parser.add_argument('--mode',
                    '-m',
                    type=str,
                    default='preprocessing',
                    help='preprocessing|train|evaluation|predict')
args = parser.parse_args()


class Runner(object):
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.model_dir = 'saved_models'

        self.hyper = Hyper(os.path.join('experiments',
                                        self.exp_name + '.json'))

        self.gpu = self.hyper.gpu
        self.preprocessor = None
        self.triplet_metrics = F1_triplet()
        self.ner_metrics = F1_ner()
        self.optimizer = None
        self.model = None
        self.opt_lr = 0.01
        self.patient = 10000
        if getattr(self.hyper, 'lr'):
            self.opt_lr = self.hyper.lr
        if getattr(self.hyper, 'patient'):
            self.patient= self.hyper.patient

    def _optimizer(self, name, model):
        m = {
        'adam': Adam(model.parameters(), lr = self.opt_lr),
        'sgd': SGD(model.parameters(), lr = self.opt_lr),
        'adamw': AdamW(model.parameters(), lr = self.opt_lr)
        }
        return m[name]

    def _init_model(self):
        if self.gpu == -1:  # no gpu
            self.model = MultiHeadSelection(self.hyper)
        else:
            self.model = MultiHeadSelection(self.hyper).cuda(self.gpu)

    def preprocessing(self):
        if self.exp_name == 'conll_selection_re':
            self.preprocessor = Conll_selection_preprocessing(self.hyper)
        elif self.exp_name == 'chinese_selection_re':
            self.preprocessor = Chinese_selection_preprocessing(self.hyper)
        elif self.exp_name == 'conll_bert_re':
            self.preprocessor = Conll_bert_preprocessing(self.hyper)
        elif self.exp_name == 'medical_selection_re':
            self.preprocessor = Medical_selection_preprocessing(self.hyper)

        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()
        self.preprocessor.gen_vocab(min_freq=1)
        # for ner only
        self.preprocessor.gen_bio_vocab()

    def run(self, mode: str):
        if mode == 'preprocessing':
            self.preprocessing()
        elif mode == 'train':
            self._init_model()
            self.optimizer = self._optimizer(self.hyper.optimizer, self.model)
            self.train()
        elif mode == 'evaluation':
            self._init_model()
            self.load_model(epoch=self.hyper.evaluation_epoch)
            self.evaluation()
        elif mode == 'predict':
            self._init_model()
            self.load_model(epoch=self.hyper.evaluation_epoch)
            self.predict()
        else:
            raise ValueError('invalid mode')

    def load_model(self, epoch: int):
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir,
                             self.exp_name + '_' + str(epoch))))

    def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, self.exp_name + '_' + str(epoch)))

    @staticmethod
    def predict_result_save(fd, text_list, results, jobid=None):
        """
        function: 将预测结果以spo格式写入文件
        text_list: tuple(str,)
        results: list(list(dict, ), )
        """
        for i in range(len(text_list)):
            elem = {'text': text_list[i], 'spo_list_predict': results[i]}
            if isinstance(jobid, tuple):
                elem['job_id'] = jobid[i]
            fd.write(json.dumps(elem, ensure_ascii=False, indent=4))
            fd.write('\n====\n')

    def evaluation(self):
        dev_set = Selection_Dataset(self.hyper, self.hyper.dev)
        loader = Selection_loader(dev_set, batch_size=self.hyper.eval_batch, pin_memory=True)
        self.triplet_metrics.reset()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        result_file_tmp = '/tmp/results_json.{}.txt'.format(datetime.strftime(datetime.now(), '%Y-%m-%d'))
        fd = codecs.open(result_file_tmp, mode='w', encoding='utf8')

        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model(sample, is_train=False)
                self.triplet_metrics(output['selection_triplets'], output['spo_gold'])
                self.ner_metrics(output['gold_tags'], output['decoded_tag'])
                self.predict_result_save(fd, sample.text, output['selection_triplets'], jobid=sample.jobid)

            triplet_result = self.triplet_metrics.get_metric()
            ner_result = self.ner_metrics.get_metric()
            print('Triplets-> ' +  ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in triplet_result.items() if not name.startswith("_")
            ]) + ' ||' + 'NER->' + ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in ner_result.items() if not name.startswith("_")
            ]))
        fd.close()
        triplet_result_detail = self.triplet_metrics.get_metric_detail()
        self.print_eval_result(triplet_result_detail)

        return triplet_result['fscore'], ner_result['fscore']

    @staticmethod
    def print_eval_result(details):
        for k, v in details.items():
            print('%s:\tp: %.4f\tr: %.4f\tf1: %.4f\tatg: %s' %(k, 
            v['precision'], v['recall'], v['fscore'], v['ABC']))

    def train(self):
        train_set = Selection_Dataset(self.hyper, self.hyper.train)
        loader = Selection_loader(train_set, batch_size=self.hyper.train_batch, pin_memory=True)

        if getattr(self.hyper, 'resume_model') and self.hyper.resume_model != 0:
            self.load_model(self.hyper.resume_model)
        
        patient_cnt = 0
        epoch_best, spo_f1_best = 0, 0.
        for epoch in range(self.hyper.epoch_num):
            self.model.train()
            pbar = tqdm(enumerate(BackgroundGenerator(loader)),
                        total=len(loader))

            for batch_idx, sample in pbar:
                self.optimizer.zero_grad()
                output = self.model(sample, is_train=True)
                loss = output['loss']
                loss.backward()
                self.optimizer.step()

                pbar.set_description(output['description'](
                    epoch, self.hyper.epoch_num))

            # 注意这里不是连续eval，而是在print_epoch间隔时做的eval
            if (epoch + 1) % self.hyper.print_epoch == 0:
                spo_f1, _ = self.evaluation()
                if spo_f1 > spo_f1_best:
                    spo_f1_best = spo_f1 
                    epoch_best = epoch + 1
                    print('====Best SPO F1 in Epoch={}===='.format(epoch_best))
                    self.save_model(epoch_best)
                    patient_cnt = 0
                else:
                    patient_cnt += 1
            if patient_cnt >= self.patient:
                break

    def predict(self):
        """
        只对text文本进行预测，无标注数据
        """
        dev_set = Selection_Dataset(self.hyper, self.hyper.test, type='predict')
        loader = Selection_loader_predict(dev_set, batch_size=self.hyper.eval_batch, pin_memory=True)
        self.triplet_metrics.reset()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        result_file_tmp = '/tmp/results_json.{}.txt'.format(datetime.strftime(datetime.now(), '%Y-%m-%d'))
        fd = codecs.open(result_file_tmp, mode='w', encoding='utf8')
        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model.predict(sample)
                self.predict_result_save(fd, sample.text, output['selection_triplets'])
        fd.close()


if __name__ == "__main__":
    runner = Runner(exp_name=args.exp_name)
    runner.run(mode=args.mode)


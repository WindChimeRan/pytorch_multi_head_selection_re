# /usr/bin/env python
# encoding=utf8
# 把LAS标注的实体关系格式转化为Baidu-InfExt的三元组格式SPO
# 增加train/dev/test样本拆分


import random
import os
import json
import codecs
import click
from tqdm import tqdm


@click.command()
@click.option('-i', 'inputfile', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('-o', 'outputfile', type=click.Path())
@click.option('-re', 'all_schemas', type=click.Path())
def trans2pos(inputfile, outputfile, all_schemas):
    out_fd = codecs.open(outputfile, mode='w', encoding='utf8')
    rel_schames = set()
    with codecs.open(inputfile, encoding='utf8') as fd:
        jdata = json.load(fd)
        for elem in tqdm(jdata, desc='processing'):
            text = elem.get('text', '')
            jobid = elem.get('job_id', 0)
            ann = elem.get('ann', [])
            spo_list, rel_schame = ann2spo(text, ann)
            elem_new = {'jobid': jobid, 'text': text, 'spo_list': spo_list}
            out_fd.write(json.dumps(elem_new, ensure_ascii=False))
            out_fd.write('\n')
            rel_schames |= rel_schame

    with codecs.open(all_schemas, mode='w', encoding='utf8') as ofd:
        for e in rel_schames:
            ofd.write(e)
            ofd.write('\n')

    # split samples to train/dev/test within 8:1:1
    result = gen_samples(outputfile)
    print('count of :\t train:{}\tdev:{}\ttest{}'.format(result[0], result[1], result[2]))


def gen_samples(outputfile):
    # train:dev:test = 8:1:1
    output = os.path.dirname(outputfile)
    train_fd = codecs.open(os.path.join(output, 'train.txt'), mode='w', encoding='utf8')
    dev_fd = codecs.open(os.path.join(output, 'dev.txt'), mode='w', encoding='utf8')
    test_fd = codecs.open(os.path.join(output, 'test.txt'), mode='w', encoding='utf8')

    train_cnt, dev_cnt, test_cnt = 0, 0, 0
    random.seed(7)
    with codecs.open(outputfile, encoding='utf8') as fd:
        for line in fd:
            val = random.randint(1, 10)
            if val == 1:  # test
                test_fd.write(line.strip())
                test_fd.write('\n')
                test_cnt += 1
            elif val == 2:  # dev
                dev_fd.write(line.strip())
                dev_fd.write('\n')
                dev_cnt += 1
            else:   # train
                train_fd.write(line.strip())
                train_fd.write('\n')
                train_cnt += 1
    train_fd.close()
    dev_fd.close()
    test_fd.close()
    return train_cnt, dev_cnt, test_cnt


def ann2spo(text, ann):
    result = []
    rel_schames = []
    words = ann.get('words', [])
    word_id_idx = word_idx(words)
    for rel in ann.get('relationList', []):
        fid, tid = rel['fromWord']['id'], rel['toWord']['id']
        rel_tag = rel['relationTag']
        fid_s, fid_e = word_id_idx[fid]['start'], word_id_idx[fid]['end']
        tid_s, tid_e = word_id_idx[tid]['start'], word_id_idx[tid]['end']
        elem = {'predicate': rel_tag,
                'subject_type': word_id_idx[fid]['entityTag'],
                'object_type':  word_id_idx[tid]['entityTag'],
                'subject': text[fid_s:fid_e],
                'object':  text[tid_s:tid_e],
                'subject_pos': [fid_s, fid_e],
                'object_pos': [tid_s, tid_e],
                }
        result.append(elem)
        rel_schames.append(json.dumps({"subject_type": word_id_idx[fid]['entityTag'],
                                       "predicate": rel_tag,
                                       "object_type": word_id_idx[tid]['entityTag'],
                                      }, ensure_ascii=False))
    return result, set(rel_schames)


def word_idx(words):
    """
    :param words: [{id:x, start:x, end:x, entityTag:x}, ]
    :return:
    """
    result = {}
    for wd in words:
        id = wd.get('id', None)
        if not id:
            raise Exception('id not in words')
        if id not in result:
            result[id] = wd
    return result


if __name__ == '__main__':
    trans2pos()

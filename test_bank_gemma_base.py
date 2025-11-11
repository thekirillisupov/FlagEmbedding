from datasets import load_dataset

import gzip
import logging
import os
import numpy as np
import tarfile
import json
import pandas as pd
from datetime import datetime
from torch.optim import AdamW 
from torch import nn
from torch import cuda
import torch

import sys
import shutil
import importlib
from statistics import mean
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from torch.utils.data import DataLoader
  
import yaml  
from collections import defaultdict

import torch
from typing import List
from FlagEmbedding import FlagLLMReranker

class PassageRanker:
    def __init__(self, path):
        self.model = FlagLLMReranker(
            path, 
            query_max_length=128,
            passage_max_length=512,
            use_fp16=True,
            devices=['cuda:0']
        )
           
    def rank(
        self,
        query: str,
        documents: List[str],
        batch_size: int = 32,
        convert_to_tensor=True, 
        show_progress_bar=False
    ) -> list[dict]:
        scores = self.model.compute_score(list(zip([query]*len(documents), documents)), normalize=True)
        results = [{"corpus_id": idx, "score": score} for idx, score in enumerate(scores)]

        return sorted(results, key=lambda x: x["score"], reverse=True)

reranker = PassageRanker(path='test_decoder_only_base_bge-reranker-v2-gemma/checkpoint-12442')

# test case
print(reranker.rank("сберспасибо", ["'Разделы:\nСберСпасибо / Базовая лояльность\nВопрос:\nЧто такое СберСпасибо\nОтвет:\nСберСпасибо – это бесплатная бонусная программа от Сбера. Чтобы получать бонусы, оплачивайте покупки картами СберБанка в выбранных категориях с кешбэком, делайте покупки у партнёров программы, участвуйте в акциях, получайте бонусы за оформление и использование продуктов банка и сервисов партнеров. Накопленные бонусы можно использовать для получения скидки у партнёров программы, обменивать на рубли и переводить другим участникам'"]))

import sys
sys.path.append('/home/jovyan/isupov/reranker')
import json  

path = '/home/jovyan/isupov/reranker'

from eval import eval_mean_metrics_question_with_rel_passages, eval_faqes_via_all_threshold, aggregate_result_for_one_model, eval_rel_faqes_via_all_threshold, eval_faq, eval_mean_metrics_wo_threshold

with open(f'{path}/data/sbol_search_bar_retrieval/annotations/v16/test/test_reranker.json', 'r') as json_file:
    test_sb = json.load(json_file)

with open(f'{path}/data/sbol_va_retrieval/annotations/v13/test/reranker_test.json', 'r') as json_file:
    test_va = json.load(json_file)

with open(f'{path}/data/sbol_search_bar_semirelevance/annotations/v11/test/reranker_test.json', 'r') as json_file:
    test_semi_sb =  json.load(json_file)

with open(f'{path}/data/sbol_va_semirelevance/annotations/v3/test/reranker_test.json', 'r') as json_file:
    test_semi_va =  json.load(json_file)

with open(f'{path}/data/ivr_semirelevance/annotations/v3/test/reranker_test.json', 'r') as json_file:
    test_ivr =  json.load(json_file)

with open(f'{path}/data/copilot_cx_semirelevance/annotations/v3/test/reranker_test.json', 'r') as json_file:
    test_cx = json.load(json_file)

with open(f'{path}/data/copilot_sc_semirelevance/annotations/v2/test/reranker_test.json', 'r') as json_file:
    test_sc = json.load(json_file)

with open(f'{path}/data/ckiz_90_ranking_semirel/annotations/v1/test/reranker_test.json', 'r') as json_file:
    test_ckiz_90 = json.load(json_file)

#with open('./data/lik_semirelevance/annotations/v1/test/reranker_test.json', 'r') as json_file:
#    test_lik =  json.load(json_file)

with open(f'{path}/data/universal_retrieval/lik_test_v1.json', 'r') as json_file:
    test_lik =  json.load(json_file)

#with open('/home/jovyan/isupov/data/index_generated.json', 'r') as json_file:
#    index_generated = json.load(json_file)

test_rel_faq =  test_semi_sb + test_sb + test_va  #  + test_semi_va\

print('semi-rel => rel')
for test, name in zip([test_rel_faq, test_ivr, test_cx, test_sc, test_ckiz_90, test_lik], ['test_rel_faq', 'test_ivr', 'test_cx', 'test_sc', 'test_ckiz_90', 'test_lik']):
    metrics = eval_mean_metrics_wo_threshold(reranker, test, is_relevant=['r', 's'])
    print(name)
    print('Recall@3 | Precision@3 | MRR_8')
    print(f"{metrics['Recall_3']} | {metrics['Precision_3']}| {metrics['MRR_10']}")
    print('---------') 


print('semi-rel => nerel')
for test, name in zip([test_rel_faq, test_ivr, test_cx, test_sc, test_ckiz_90], ['test_rel_faq', 'test_ivr', 'test_cx', 'test_sc', 'test_ckiz_90', 'test_lik']):
    metrics = eval_mean_metrics_wo_threshold(reranker, test, is_relevant=['r'])
    print(name)
    print('Recall@3 | Precision@3 | MRR_8')
    print(f"{metrics['Recall_3']} | {metrics['Precision_3']}| {metrics['MRR_10']}")
    print('---------')
        

"""
test_decoder_only_base_bge-reranker-v2-gemma/checkpoint-12442

test_rel_faq
Recall@3 | Precision@3 | MRR_8
0.858 | 0.579| 0.935

test_ivr
Recall@3 | Precision@3 | MRR_8
0.848 | 0.557| 0.953

test_cx
Recall@3 | Precision@3 | MRR_8
0.803 | 0.611| 0.936

test_sc
Recall@3 | Precision@3 | MRR_8
0.93 | 0.465| 0.871

test_ckiz_90
Recall@3 | Precision@3 | MRR_8
0.938 | 0.499| 0.983

test_lik
Recall@3 | Precision@3 | MRR_8
0.665 | 0.403| 0.691
"""
import torch
import numpy as np
from transformers import AutoTokenizer
import csv
test_name = 'HYM_test_dataset.csv'
device = 'cuda'
output_name = 'bert_output.csv'
data_files = {"train": test_name, "val": "HYM_val_dataset.csv"}
path = r'D:\study inf\nlp\zidonghuasuo\F_E_Network_master\automatic-pattern-select-moudle\double_tuple_extraction'
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
n_best_size = 3


def FENETWORK(model_,context):
    inputs = tokenizer(
        '[HYM]',
        context,
        max_length=120,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")
    inputs1 = {}
    for k, v in inputs.items():
        v = np.array(v)
        v = np.expand_dims(v, axis=0)
        v = torch.tensor(v)
        inputs1[k] = v
    inputs = {k: v.to(device) for k, v in inputs1.items()}
    model_.to(device)

    outputs = model_(inputs)
    context_start, context_end, query_start, query_end = torch.chunk(outputs, 4, dim=-1)

    context_start = torch.squeeze(context_start, -1)
    context_end = torch.squeeze(context_end, -1)
    query_start = torch.squeeze(query_start, -1)
    query_end = torch.squeeze(query_end, -1)

    context_1_logits = context_start.detach().cpu().numpy()
    context_2_logits = context_end.detach().cpu().numpy()
    query_1_logits = query_start.detach().cpu().numpy()
    query_2_logits = query_end.detach().cpu().numpy()
    num_id = -1
    for context_1_logit, context_2_logit, query_1_logit, query_2_logit in zip(context_1_logits, context_2_logits,
                                                                              query_1_logits, query_2_logits):
        decode_id_o_heads = []
        decode_id_o_contexts = []

        num_id += 1
        query_1_index = np.argsort(query_1_logit)[-1: -n_best_size - 1: -1].transpose().tolist()
        query_2_index = np.argsort(query_2_logit)[-1: -n_best_size - 1: -1].transpose().tolist()
        context_1_index = np.argsort(context_1_logit)[-1: -n_best_size - 1: -1].transpose().tolist()
        context_2_index = np.argsort(context_2_logit)[-1: -n_best_size - 1: -1].transpose().tolist()
        hypernym =[]
        hypernym_score =[]
        head =[]
        head_score =[]
        if 0 in context_1_index and 0 in context_2_index:
            context_1_index.remove(0)
            context_2_index.remove(0)
            hypernym.append('NONE')
            hypernym_score.append(context_1_logit[0] + context_2_logit[0])
        if 0 in context_1_index: context_1_index.remove(0)
        if 0 in context_2_index: context_2_index.remove(0)
        for end_index in context_2_index:
            for start_index in context_1_index:
                if (end_index < start_index):
                    continue
                start_char = offset_mapping[start_index][0]
                end_char = offset_mapping[end_index][1]
                hypernym.append(context[start_char: end_char])
                hypernym_score.append(context_1_logit[start_index] + context_2_logit[end_index])
        if 0 in query_2_index and 0 in query_1_index:
            query_2_index.remove(0)
            head.append('NONE')
            head_score.append(query_1_logit[0] + query_2_logit[0])
        if 0 in query_2_index: query_2_index.remove(0)
        if 0 in query_1_index: query_1_index.remove(0)

        for end_index in query_2_index:
            for start_index in query_1_index:
                if (end_index < start_index):
                    continue
                start_char = offset_mapping[start_index][0]
                end_char = offset_mapping[end_index][1]
                head.append(context[start_char: end_char])
                head_score.append(query_1_logit[start_index] + query_2_logit[end_index])
        hypernym_dicts =[]
        head_dicts = []
        for hypernym_, hypernym_score_ in zip (hypernym,hypernym_score):
            hypernym_dicts.append({'hypernym':hypernym_,'hypernym_score':hypernym_score_})
        hypernym_dicts = sorted(hypernym_dicts, key=lambda x: x['hypernym_score'], reverse=True)
        for head_, head_score_ in zip (head,head_score):
            head_dicts.append({'head':head_,'head_score':head_score_})
        # All results
        hypernym_dicts = sorted(hypernym_dicts, key=lambda x: x['hypernym_score'], reverse=True)
        head_dicts = sorted(head_dicts, key=lambda x: x['head_score'], reverse=True)
        sum_score = 0
        num_a = 0
        num_b = 0
        for a, b in zip (head,head_score):
            for c,d in zip(hypernym,hypernym_score):
                if not (a == 'NONE') ^ (c == 'NONE'):
                    if sum_score < b+d:
                        sum_score = b+d
                        num_a = head_score.index(b)
                        num_b = hypernym_score.index(d)
        head_ = head[num_a]
        head_score_ = head_score[num_a]
        hypernym_ = hypernym[num_b]
        hypernym_score_ = hypernym_score[num_b]

        print('###输入可以在%s查看'%output_name)
        with open(output_name, "a", encoding="utf-8", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([context,hypernym_,hypernym_score_,head,head_score])
            model_out = {'context':context,
                         'hypernym':hypernym_,
                         'hypernym_score':hypernym_score_,
                         'head':head_,
                         'head_score':head_score_}
    return model_out
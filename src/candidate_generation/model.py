#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# @Author  :  xuzf
# @Contact :  xuzhengfei-email@qq.com
# @Create  :  2023-05-26 09:32:55
# @Update  :  2023-11-21 12:28:44
# @Desc    :  None
# =============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME


class Encoder(nn.Module):
    def __init__(self, model, output_dim=None):
        super(Encoder, self).__init__()
        # self.device = model.embeddings.word_embeddings.weight.device
        self.embedding_hdim = model.embeddings.word_embeddings.weight.size(1)
        self.addition_linear = None
        self.output_dim = self.embedding_hdim
        
        self.encoder = model
        if output_dim is not None:
            self.dropout = nn.Dropout(0.1)
            self.addition_linear = nn.Linear(self.embedding_hdim, output_dim)
            self.output_dim = output_dim
        
    def forward(self, input_ids):
        output = self.encoder(*to_bert_input(input_ids))
        # get embedding of [CLS] token
        emb = output[0][:, 0, :]
        if self.addition_linear is not None:
            emb = self.addition_linear(self.dropout(emb))
        return emb


class BiEncoder(nn.Module):
    def __init__(self, params):
        super(BiEncoder, self).__init__()
        mention_model = BertModel.from_pretrained(params["model_name"], local_files_only=True)
        entity_model = BertModel.from_pretrained(params["model_name"], local_files_only=True)
        self.mention_encoder = Encoder(mention_model, output_dim=params["dim"])
        self.entity_encoder = Encoder(entity_model, output_dim=params["dim"])
        self.config = mention_model.config

    def forward(self, mention_ids, entity_ids):
        mention_emb = None
        if mention_ids is not None:
            mention_emb = self.mention_encoder(mention_ids)
        entity_emb = None
        if entity_ids is not None:
            entity_emb = self.entity_encoder(entity_ids)
        return mention_emb, entity_emb


class Retriever(nn.Module):
    def __init__(self, params):
        super(Retriever, self).__init__()
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(params["model_name"], files_only=True)
        
        self.biencoder = BiEncoder(params)
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)
        self.biencoder = self.biencoder.to(self.device)

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.biencoder.load_state_dict(state_dict)
    
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(self.biencoder.state_dict(), output_model_file)
        self.biencoder.config.to_json_file(output_config_file)
    
    def encode_mention(self, mention_ids):
        mention_emb, _ = self.biencoder(mention_ids, None)
        return mention_emb.cpu().detach()

    def encode_entity(self, entity_ids):
        _, entity_emb = self.biencoder(None, entity_ids)
        return entity_emb.cpu().detach()

    def score_candidate(
            self,
            mention_ids,
            entity_ids,
            random_negs=True,
            cand_pool_emb=None):
        # given pre-computed candidate encoding, usually for evaluation
        if cand_pool_emb is not None:
            mention_emb, _ = self.biencoder(mention_ids, None)
            return mention_emb.mm(cand_pool_emb.t())
        
        # train time
        if random_negs:
            # select the in-batch other entitys as negative samples
            mention_emb, entity_emb = self.biencoder(mention_ids, entity_ids)
            # scores: batch_size * batch_size
            scores = mention_emb.mm(entity_emb.t())
        else:
            # train with hard candidates
            # flatten the "negative sample num" dimension
            cand_num = entity_ids.size(1)
            flatten_shape = (entity_ids.size(0) * entity_ids.size(1), entity_ids.shape[2])
            mention_emb, entity_emb = self.biencoder(
                mention_ids, 
                entity_ids.reshape(flatten_shape))
            # mention_emb: batch_size * dim * 1
            mention_emb = mention_emb.unsqueeze(2)
            # entity_emb: batch_size * negative_sample_num * dim
            entity_emb = entity_emb.reshape(-1, cand_num, entity_emb.size(1))
            # scores: batch_size * negative_sample_num
            scores = torch.bmm(entity_emb, mention_emb).squeeze(2)
        return scores
    
    def forward(
            self, 
            mention_ids, 
            entity_ids,
            label_input=None):
        if label_input is None:
            scores = self.score_candidate(mention_ids, entity_ids)
            batch_size = scores.size(0)
            target = torch.arange(batch_size, dtype=torch.long, device=self.device)
            loss = F.cross_entropy(scores, target, reduction="mean")
        else:
            scores = self.score_candidate(mention_ids, entity_ids, random_negs=False)
            target = F.one_hot(label_input.long(), num_classes=self.params['top_k']).float()
            loss = F.binary_cross_entropy_with_logits(scores, target, reduction="mean")
        return loss, scores


def to_bert_input(token_idx, null_idx=0):
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, mask.int()
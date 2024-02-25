#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# @Author  :  xuzf
# @Contact :  xuzhengfei-email@qq.com
# @Create  :  2023-05-26 11:39:00
# @Update  :  2023-10-17 08:29:26
# @Desc    :  None
# =============================================================================
import os
import torch
import wandb
import random
import time
import numpy as np
from tqdm import tqdm, trange

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule
from transformers import WEIGHTS_NAME
from transformers import AdamW

import sys
root_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(root_dir)

from utils.params import Parser
from src.candidate_generation.model import Retriever
from src.candidate_generation.tokenization import process_mention_data

from utils.logger import get_logger
from utils.io import read_dataset, write_to_file, save_model, dump_json

logger = None

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels), outputs == labels

def evaluate(retriever, eval_dataloader, params, device, logger):
    """
    The evaluate function during training uses in-batch negatives:
    for a batch of size B, the labels from the batch are used as label candidates
    B is controlled by the parameter eval_batch_size
    """
    retriever.biencoder.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    for batch in iter_:
        batch = tuple(t.to(device) for t in batch)
        # context_input, candidate_input, _, _ = batch
        mention_input, candidate_input, label_ids = batch

        with torch.no_grad():
            if not params['just_hard']:
                _, logits = retriever(mention_input, candidate_input)
                # Using in-batch negatives, the label ids are diagonal
                label_ids = torch.LongTensor(torch.arange(mention_input.size(0))).numpy()
            else:
                _, logits = retriever(mention_input, candidate_input, label_ids)
                label_ids = torch.LongTensor(label_ids.cpu()).numpy()

        logits = logits.detach().cpu().numpy()
        tmp_eval_accuracy, _ = accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += mention_input.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    results["normalized_accuracy"] = normalized_eval_accuracy
    torch.cuda.empty_cache()
    return results


def get_optimizer(model, learning_rate):
    """ Optimizes the network with AdamWithDecay
    """
    parameters_with_decay = []
    parameters_with_decay_names = []
    parameters_without_decay = []
    parameters_without_decay_names = []
    no_decay = ['bias', 'gamma', 'beta']

    for n, p in model.named_parameters():
        if any(t in n for t in no_decay):
            parameters_without_decay.append(p)
            parameters_without_decay_names.append(n)
        else:
            parameters_with_decay.append(p)
            parameters_with_decay_names.append(n)

    optimizer_grouped_parameters = [
        {'params': parameters_with_decay, 'weight_decay': 0.01},
        {'params': parameters_without_decay, 'weight_decay': 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=learning_rate, 
        correct_bias=True
    )
    return optimizer


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def read_and_tokenize_data(dataset_name, tokenizer, params, logger):
    train_set_num = None
    if 'train' in dataset_name:
        train_set_num = params['train_steps_per_epoch'] * params["train_batch_size"]
    samples = read_dataset(dataset_name, params["data_path"], train_size_num=train_set_num)
    logger.info("Read %d samples from %s" % (len(samples), dataset_name))
    dataset, tensor_data = process_mention_data(
        samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )
    return dataset, tensor_data


# def read_hard_sample(fname, debug=False):
#     samples = torch.load(fname)
#     tensor_data = list(zip(samples["mention_vecs"], samples["context_vecs"],
#                            samples["candidate_vecs_title"], samples["candidate_vecs_desc"],
#                            samples["labels"]))
#     if debug:
#         tensor_data = tensor_data[:200]
#     return tensor_data


def main(params):
    model_output_path = os.path.join(params["output_path"], 'train_retriever_be@%s' % (time.strftime('%m-%d-%H:%M', time.localtime())))
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = get_logger(os.path.basename(__file__).split('.')[0], output_dir=model_output_path, stdout=False)

    # Init model
    retriever = Retriever(params)
    tokenizer = retriever.tokenizer
    model = retriever.biencoder
    device = retriever.device
    
    if params['use_wandb']:
        logger.info(f"Start training with Weights and Biases... As Train_Biencoder")
        wandb.init(project="wdel_train_cg", name=model_output_path, config=params)
        wandb.watch(retriever)

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            params["gradient_accumulation_steps"]))

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps
    params["train_batch_size"] = (params["train_batch_size"] // params["gradient_accumulation_steps"])
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load train and eval data
    if not params["just_hard"]:
        # _, train_tensor_data = read_and_tokenize_data("hansel-train", tokenizer, params, logger)
        # torch.save(train_tensor_data, "/home/xuzf/workspace/el/wdel/data/hansel/train_tensor_data_cg_6400000.pt")
        # train_tensor_data = torch.load("/home/xuzf/workspace/el/wdel/data/hansel/train_tensor_data_cg_3200000.pt")
        train_tensor_data = torch.load("/home/xuzf/workspace/el/wdel/data/hansel/train_tensor_data_cg_6400000.pt")
        if not params['debug']:
            _, valid_tensor_data = read_and_tokenize_data("hansel-dev", tokenizer, params, logger)
        else:
            valid_tensor_data = train_tensor_data
    # else:
        # assert params["train_hard_sample_path"] and params["eval_hard_sample_path"], "Error! hard sample path is empty."
        # train_tensor_data = read_hard_sample(params["train_hard_sample_path"], params['debug'])
        # valid_tensor_data = read_hard_sample(params["eval_hard_sample_path"], params['debug'])

    train_sampler = RandomSampler(train_tensor_data) if params["shuffle"] else SequentialSampler(train_tensor_data)
    train_dataloader = DataLoader(train_tensor_data, sampler=train_sampler, batch_size=train_batch_size)
    
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size)

    # evaluate before training
    results = evaluate(retriever, valid_dataloader, params, device, logger)

    time_start = time.time()

    # save the parameters to file
    dump_json(os.path.join(model_output_path, "training_params.json"), params, indent=4, sort_keys=True)

    logger.info("Starting training")

    # optimizer
    optimizer = get_optimizer(model, params['learning_rate'])
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)

    model.train()

    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = params["num_train_epochs"]
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        results = None

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            mention_input, candidate_input, labels_input = batch
            if not params['just_hard']:
                loss, _ = retriever(mention_input, candidate_input)
            else:
                loss, _ = retriever(mention_input, candidate_input, labels_input)

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info("Step {} - epoch {} average loss: {}".format(
                    step,
                    epoch_idx,
                    tr_loss / (params["print_interval"] * grad_acc_steps),
                ))
                if params['use_wandb']:
                    wandb.log({
                        "loss": tr_loss / (params["print_interval"] * grad_acc_steps),
                        "lr": optimizer.state_dict()['param_groups'][0]['lr'],
                        "print_step": step,
                    })
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                results = evaluate(retriever, valid_dataloader, params, device, logger)
                model.train()
                if params["use_wandb"]:
                    wandb.log(
                        {
                            "accuracy": results["normalized_accuracy"] * 100,
                            "eval_step": step
                        }
                    )

        if not params['debug']:
            logger.info("***** Saving fine - tuned model *****")
            epoch_output_folder_path = os.path.join(model_output_path, "epoch_{}".format(epoch_idx))
            save_model(model, tokenizer, epoch_output_folder_path)

        results = evaluate(retriever, valid_dataloader, params, device, logger)
        if params["use_wandb"]:
            wandb.log(
                {
                    "accuracy_epoch": results["normalized_accuracy"] * 100,
                    "epoch_step": step
                }
            )


        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]

    execution_time = (time.time() - time_start) / 60
    write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    if not params['debug']:
        params["path_to_model"] = os.path.join(
            model_output_path,
            "epoch_{}".format(best_epoch_idx),
            WEIGHTS_NAME,
        )
        retriever = Retriever(params)
        save_model(retriever.biencoder, tokenizer, model_output_path)


if __name__ == "__main__":
    parser = Parser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)

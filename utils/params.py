#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# @Author  :  xuzf
# @Contact :  xuzhengfei-email@qq.com
# @Create  :  2023-05-31 16:57:21
# @Update  :  2023-10-18 14:07:38
# @Desc    :  None
# =============================================================================
import argparse


class Parser(argparse.ArgumentParser):
    """
    Provide an opt-producer and CLI arguement parser.

    More options can be added specific by paassing this object and calling
    ''add_arg()'' or add_argument'' on it.

    :param add_base_args:
        (default True) initializes the default arguments for biencoder & retriever package.
    :param add_model_args:
        (default False) initializes the default arguments for loading models,
        including initializing arguments from the model.
    """

    def __init__(
        self,
        add_base_args=True,
        add_model_args=False,
        description='biencoder parser',
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            formatter_class=argparse.HelpFormatter,
            add_help=add_base_args,
        )
        # self.blink_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        # os.environ['BLINK_HOME'] = self.blink_home

        self.add_arg = self.add_argument

        self.overridable = {}

        if add_base_args:
            self.add_base_args()
        if add_model_args:
            self.add_model_args()

    def add_base_args(self, args=None):
        """
        Add base args across all scripts.
        """
        parser = self.add_argument_group("Common Arguments")
        parser.add_argument("--silent", action="store_true", help="Whether to print progress bars.")
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Whether to run in debug mode with only 200 samples.",
        )
        parser.add_argument(
            "--no_cuda",
            action="store_true",
            help="Whether not to use CUDA when available",
        )
        parser.add_argument("--top_k", default=10, type=int)
        parser.add_argument("--seed", type=int, default=52313, help="random seed for initialization")
        parser.add_argument(
            "--use_wandb",
            action='store_true',
            help="Whether use the wandb to monitor.",
        )
        parser.add_argument("--comment", type=str, default="", help="desc for this run")

    def add_model_args(self, args=None):
        """
        Add model args.
        """
        parser = self.add_argument_group("Model Arguments")
        parser.add_argument(
            "--max_cot_length",
            default=256,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_context_length",
            default=128,
            type=int,
            help="The maximum total context input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_cand_length",
            default=128,
            type=int,
            help="The maximum total label input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--path_to_model",
            default=None,
            type=str,
            required=False,
            help="The full path to the model to load.",
        )
        parser.add_argument(
            "--save_tensor_data",
            action='store_true',
            help="Whether save the tensor_dataset.",
        )
        parser.add_argument(
            "--use_LLM",
            action='store_true',
            help="Whether use the LLM to boost reranker.",
        )
        parser.add_argument(
            "--model_name",
            default="bert-base-chinese",
            type=str,
            help="Bert pre-trained model selected in the list: bert-base-uncased, "
            "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
        )
        parser.add_argument(
            "--dim",
            type=int,
            default=None,
            help="Output dimention of bi-encoders.",
        )
        parser.add_argument(
            "--data_path",
            default="data/dataset",
            type=str,
            help="The path to the train data.",
        )
        parser.add_argument(
            "--output_path",
            default=None,
            type=str,
            required=True,
            help="The output directory where generated output file (model, etc.) is to be dumped.",
        )

    def add_training_args(self, args=None):
        """
        Add model training args.
        """
        parser = self.add_argument_group("Model Training Arguments")
        parser.add_argument("--just_hard", action="store_true", help="Train the model with hard negative sample.")
        parser.add_argument("--train_hard_sample_path", default=None, type=str, help="Path for hard negative sample")
        parser.add_argument("--eval_hard_sample_path", default=None, type=str, help="Path for hard negative sample")
        parser.add_argument("--reranker_train_file", default=None, type=str, help="Path for reranker train sample")
        parser.add_argument("--reranker_dev_file", default=None, type=str, help="Path for reranker dev sample")
        parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument(
            "--learning_rate",
            default=3e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--num_train_epochs",
            default=1,
            type=int,
            help="Number of training epochs.",
        )
        parser.add_argument(
            "--train_steps_per_epoch",
            default=10000,
            type=int,
            help="Number of training steps per epoch.",
        )
        parser.add_argument(
            "--print_interval",
            type=int,
            default=10,
            help="Interval of loss printing",
        )
        parser.add_argument(
            "--eval_interval",
            type=int,
            default=100,
            help="Interval for evaluation during training",
        )
        parser.add_argument("--save_interval", type=int, default=1, help="Interval for model saving")
        parser.add_argument(
            "--warmup_proportion",
            default=0.1,
            type=float,
            help="Proportion of training to perform linear learning rate warmup for. "
            "E.g., 0.1 = 10% of training.",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumualte before performing a backward/update pass.",
        )
        parser.add_argument(
            "--shuffle",
            type=bool,
            default=False,
            help="Whether to shuffle train data",
        )

    def add_eval_args(self, args=None):
        """
        Add model evaluation args.
        """
        parser = self.add_argument_group("Model Evaluation Arguments")
        parser.add_argument("--reranker_test_file", default=None, type=str, help="Path for reranker test sample")
        parser.add_argument(
            "--eval_batch_size",
            default=8,
            type=int,
            help="Total batch size for evaluation.",
        )
        parser.add_argument(
            "--mode",
            default="valid",
            type=str,
            help="train / dev / test",
        )
        parser.add_argument(
            "--save_topk_result",
            action="store_true",
            help="Whether to save prediction results.",
        )
        parser.add_argument(
            "--save_rerank_result",
            action="store_true",
            help="Whether to save rerank results.",
        )
        parser.add_argument("--encode_batch_size", default=8, type=int, help="Batch size for encoding.")
        parser.add_argument(
            "--cand_pool_path",
            default=None,
            type=str,
            help="Path for cached candidate pool (id tokenization of candidates)",
        )
        parser.add_argument(
            "--cand_encode_path",
            default=None,
            type=str,
            help="Path for cached candidate encoding",
        )
        parser.add_argument(
            "--entity_dict_path",
            default=None,
            type=str,
            help="Path for entity dict (dict[title, text])"
        )

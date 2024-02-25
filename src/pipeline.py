import os
import sys
import json
from typing import List, Dict
import mysql.connector
from flask import Flask, request, jsonify

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_dir)

from utils.io import load_pickle
from src.candidate_generation.model import Retriever
from src.candidate_generation.tokenization import entity_text_modeling
from src.candidate_generation.prediction import get_topk_candidates_faiss, load_faiss_index
from src.entity_disambiguation.prediction import load_lora_model, inference

app = Flask(__name__)

# 候选生成信息
cg_config = {
    "model_name": "/home/xuzf/models/bert-base-multilingual-uncased",
    "path_to_model": "output/cg/train_retriever_be@01-27-12:40/pytorch_model.bin",
    "hnsw_index_path": "output/cg/train_retriever_be@01-27-12:40/hnsw32_efC512_index.faiss",
    "kb_idx2qid_path": "output/cg/train_retriever_be@01-27-12:40/wk_info_all_pool_idx2qid.pkl",
}

# MySQL数据库连接池信息
pool_config = {
    "pool_name": "wikidata_pool",
    "pool_size": 5,
    "host": "127.0.0.1",
    "user": os.environ.get('DB_USER'),
    "password": os.environ.get('DB_PASSWORD'),
    "database": "wikidata",
}

# 实体消歧信息
ed_config = {
    "peft_model_path": "output/lora/train_lora_reranker@02-11-13:50/checkpoint-70000",
}


def init_environment():
    global retriever
    global idx2qid
    global index_faiss

    # 加载实体检索模型
    params = {
        "no_cuda": False,
        "model_name": cg_config["model_name"],
        "path_to_model": cg_config["path_to_model"],
        "dim": 128,
        "top_k": 10,
    }
    retriever = Retriever(params=params)
    # 加载faiss索引
    index_faiss = load_faiss_index(cg_config["hnsw_index_path"])
    # 加载 idx2qid
    idx2qid = load_pickle(cg_config["kb_idx2qid_path"])
    print("CG environment initialized.")

    # 创建 MySQL 连接池
    global cnxpool
    cnxpool = mysql.connector.pooling.MySQLConnectionPool(**pool_config)
    print("MySQL connection pool initialized.")

    # 加载消歧模型
    global model
    global tokenizer
    model, tokenizer, _ = load_lora_model(ed_config["peft_model_path"])
    print("ED environment initialized.")

init_environment()

def candidate_generation(doc: str, mentions: List[Dict]):
    """
    候选生成
    :param doc: 文档 (str)
    :param mentions: 实体列表 (List[Dict]), 格式为 [{"mention": "xxx", "start": 10, "end": 15}]
    :return: 候选列表 (List[List]) , 格式为 [["Q1", "Q2", "Q3"], ["Q4", "Q5", "Q6"]]
    """
    # 构造查询
    query_list = []
    for mention in mentions:
        query = {}

        query["mention"] = mention["mention"]
        query["context_left"] = doc[: mention["start"]]
        query["context_right"] = doc[mention["end"] :]
        query_list.append(query)

    # 候选生成
    topk_candidates = get_topk_candidates_faiss(retriever, query_list, idx2qid, index_faiss, top_k=10)

    return topk_candidates


def entity_disambiguation(
    doc: str,
    mentions: List[Dict],
    candidates: List[List[Dict]],
    max_context_length=128,
    max_desc_length=256,
    batch_size=10,
    prompt_text='判断文本中用 "<<" 和 ">>" 标记起始和终止位置的指称是否指代给定的实体。如果提及指代给定实体则输出"是"，否则输出"否"。',
    silent=True,
):
    """
    候选生成
    :param doc: 文档 (str)
    :param mentions: 实体列表 (List[Dict]), 格式为 [{"mention": "xxx", "start": 10, "end": 15}]
    :param candidates: 候选列表 (List[List[Dict]]), 格式为 [[{"qid": "Q1", "label":{"zh": "xxx", "en": "xxx"}, ...}]]
    :return: 重排序的候选列表
    """
    # 构造查询
    query_list = []
    for mention_id, mention_info in enumerate(mentions):
        mention = mention_info["mention"]
        context_left = doc[: mention_info["start"]]
        context_right = doc[mention_info["end"] :]

        mention_length = len(mention) + 4
        context_left_quota = max_context_length - mention_length // 2
        context_right_quota = max_context_length - mention_length - context_left_quota
        context = context_left[-context_left_quota:] + f"<<{mention}>>" + context_right[:context_right_quota]

        for candidate_id, cand in enumerate(candidates[mention_id]):
            _, cand_label, cand_desc = entity_text_modeling(cand)
            inputs = (
                prompt_text
                + f"\n【文本】：{context}\n【指称】：{mention}\n【实体】：{cand_label}\n【实体描述】：{cand_desc[:max_desc_length]}"
            )
            query_list.append({"query_id": f"{mention_id}-{candidate_id}", "inputs": inputs})
    # 实体消歧
    results = inference(model, tokenizer, query_list, batch_size=batch_size, silent=silent)

    # 结果处理
    reranked_candidates = []
    for mention_id in range(len(mentions)):
        scores = results[mention_id]["scores"]
        for candidate_id in range(len(candidates[mention_id])):
            candidates[mention_id][candidate_id]["score"] = scores[candidate_id]
        # 候选排序
        sorted_cands = sorted(candidates[mention_id], key=lambda x: x["score"], reverse=True)
        reranked_candidates.append(sorted_cands)

    return reranked_candidates


def query_data(qid_list):
    try:
        # 从连接池获取连接
        connection = cnxpool.get_connection()
        cursor = connection.cursor()

        # 构建 IN 子句的字符串
        qid_string = ", ".join(["%s" for _ in qid_list])

        # 使用 LEFT JOIN 查询 Entity 表、P31 表和 P279 表的信息
        cursor.execute(
            f"""
            SELECT
                e.qid,
                e.label_en,
                e.label_zh,
                e.desc_en,
                e.desc_zh,
                e.alt_en,
                e.alt_zh,
                p31.p31_qid,
                p31.p31_en,
                p31.p31_zh,
                p279.p279_qid,
                p279.p279_en,
                p279.p279_zh
            FROM
                Entity e
            LEFT JOIN
                P31 p31 ON e.id = p31.entity_id
            LEFT JOIN
                P279 p279 ON e.id = p279.entity_id
            WHERE
                e.qid IN ({qid_string})
        """,
            tuple(qid_list),
        )

        rows = cursor.fetchall()

        results = {}

        for qid in qid_list:
            # 以 qid 为键构建结果字典
            results[qid] = {
                "qid": None,
                "label": {"en": None, "zh": None},
                "desc": {"en": None, "zh": None},
                "alt": {"en": [], "zh": []},
                "P31": {"qid": [], "en": [], "zh": []},
                "P279": {"qid": [], "en": [], "zh": []},
            }

        for row in rows:
            qid = row[0]

            if results[qid]["qid"] is None:
                # 处理 qid 列
                results[qid]["qid"] = qid
                # 处理 label 列
                label_en = row[1]
                label_zh = row[2]
                if label_en:
                    results[qid]["label"]["en"] = label_en
                if label_zh:
                    results[qid]["label"]["zh"] = label_zh
                # 处理 desc 列
                desc_en = row[3]
                desc_zh = row[4]
                if desc_en:
                    results[qid]["desc"]["en"] = desc_en
                if desc_zh:
                    results[qid]["desc"]["zh"] = desc_zh
                # 处理 alt 列
                alt_en = row[5]
                alt_zh = row[6]
                if alt_en:
                    results[qid]["alt"]["en"] = json.loads(alt_en)
                if alt_zh:
                    results[qid]["alt"]["zh"] = json.loads(alt_zh)
            # 处理 P31 列
            p31_qid = row[7]
            p31_en = row[8]
            p31_zh = row[9]
            if p31_qid:
                results[qid]["P31"]["qid"].append(json.loads(p31_qid))
            if p31_en:
                results[qid]["P31"]["en"].append(json.loads(p31_en))
            if p31_zh:
                results[qid]["P31"]["zh"].append(json.loads(p31_zh))
            # 处理 P279 列
            p279_qid = row[10]
            p279_en = row[11]
            p279_zh = row[12]
            if p279_qid:
                results[qid]["P279"]["qid"].append(json.loads(p279_qid))
            if p279_en:
                results[qid]["P279"]["en"].append(json.loads(p279_en))
            if p279_zh:
                results[qid]["P279"]["zh"].append(json.loads(p279_zh))
        return results

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return {"error": str(err)}

    finally:
        # 关闭 cursor，将连接返回到连接池
        if "cursor" in locals():
            cursor.close()


@app.route("/cg", methods=["POST"])
def cg_query_handler():
    try:
        data = request.json
        doc = data["doc"]
        mentions = data.get("mentions", [])

        if len(mentions) < 1:
            return jsonify({"error": "Empty qid_list"})

        results = candidate_generation(doc, mentions)
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/query_wd", methods=["POST"])
def wd_query_handler():
    try:
        data = request.json
        qid_list = data.get("qid_list", [])

        if not qid_list:
            return jsonify({"error": "Empty qid_list"})

        results = query_data(qid_list)
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/ed", methods=["POST"])
def ed_query_handler():
    try:
        data = request.json
        doc = data["doc"]
        mentions = data.get("mentions", [])
        candidates = data.get("candidates", [])

        if len(mentions) < 1 or len(candidates) < 1:
            return jsonify({"error": "Empty mentions or candidates"})

        results = entity_disambiguation(doc, mentions, candidates)
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/el", methods=["POST"])
def el_query_handler():
    try:
        data = request.json
        doc = data["doc"]
        mentions = data.get("mentions", [])
        # 候选实体生成
        candidate_qids_list = candidate_generation(doc, mentions)
        # 查询实体信息
        all_qids = [q for qids in candidate_qids_list for q in qids]
        all_qids = list(set(all_qids))
        candidates_info = query_data(all_qids)
        candidates = []
        for qids in candidate_qids_list:
            cands = []
            for qid in qids:
                cand_info = candidates_info[qid]
                cands.append(cand_info)
            candidates.append(cands)
        # 实体消歧
        reranked_candidates = entity_disambiguation(doc, mentions, candidates, silent=False)
        print("Entity disambiguation done.")
        return jsonify(reranked_candidates)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

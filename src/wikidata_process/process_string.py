import concurrent.futures
from hanziconv import HanziConv


def handle_qid(line):
    line = line.lstrip("<http://www.wikidata.org/entity/").strip().rstrip(">") + "\n"
    return line


def handle_qid_text(line):
    qid, text = line.strip().split("\t")
    qid = qid.lstrip("<http://www.wikidata.org/entity/").strip().rstrip(">")
    if text.endswith("@zh"):
        text = HanziConv.toSimplified(text[:-3]) + "@zh"
    return qid + "\t" + text + "\n"


def handle_qid_qid(line):
    qid1, qid2 = line.strip().split("\t")
    qid1 = qid1.lstrip("<http://www.wikidata.org/entity/").strip().rstrip(">")
    qid2 = qid2.lstrip("<http://www.wikidata.org/entity/").strip().rstrip(">")
    return qid1 + "\t" + qid2 + "\n"

def handle_qids(line):
    # 初始化一个列表来存放字段的起始位置
    field_positions = [-1]

    # 获取每个\t的位置
    while True:
        index = line.find("\t", field_positions[-1] + 1)
        if index == -1:
            break
        field_positions.append(index)

    # 获取每个字段的值
    fields =[]
    for i in range(len(field_positions)):
        if i + 1 < len(field_positions):
            fields.append(line[field_positions[i] + 1 : field_positions[i + 1]])
        else:
            fields.append(line[field_positions[i] + 1 :])

    # 获取每个字段的值
    normal_fields = []
    for qid in fields:
        normal_fields.append(qid.lstrip("<http://www.wikidata.org/entity/").strip().rstrip(">"))

    return "\t".join(normal_fields) + '\n'

def handle_qid_text_text_text_text(line):
    # 初始化一个列表来存放字段的起始位置
    field_positions = [-1]

    # 获取每个\t的位置
    while True:
        index = line.find("\t", field_positions[-1] + 1)
        if index == -1:
            break
        field_positions.append(index)

    # 获取每个字段的值
    fields =[]
    for i in range(len(field_positions)):
        if i + 1 < len(field_positions):
            fields.append(line[field_positions[i] + 1 : field_positions[i + 1]])
        else:
            fields.append(line[field_positions[i] + 1 :])

    # 获取每个字段的值
    value1, value2, value3, value4, value5 = fields
    value1 = value1.lstrip("<http://www.wikidata.org/entity/").strip().rstrip(">")
    if value4.endswith('@zh'):
        value4 = HanziConv.toSimplified(value4[:-3]) + "@zh"
    if value5.endswith('@zh'):
        value5 = HanziConv.toSimplified(value5[:-3]) + "@zh"

    return value1 + "\t" + value2 + "\t" + value3 + "\t" + value4 + "\t" + value5 + "\n"


def process_file_chunk(chunk):
    return [handle_qid_qid(line) for line in chunk]


def process_file(in_fpath, out_fpath, num_workers=4):
    results = []

    with open(in_fpath, "r") as file:
        lines = file.readlines()

    chunk_size = (len(lines) + num_workers - 1) // num_workers
    chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file_chunk, chunk) for chunk in chunks]

        # 等待所有任务完成
        concurrent.futures.wait(futures)

        # 按照原文件的顺序合并结果
        for future in futures:
            results.extend(future.result())

    # 将结果写入文件
    with open(out_fpath, "w") as output:
        output.writelines(results)


# 调用示例
in_fpath = "KB/wikidata-2023-12-22/query_result/sameAs.tsv"
out_fpath = "KB/wikidata-2023-12-22/tmp/sameAs.txt"
process_file(in_fpath, out_fpath=out_fpath, num_workers=20)

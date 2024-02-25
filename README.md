# WDEL: 基于Wikidata的实体链接系统

WDEL是一个基于多语言双编码器的候选实体生成模型和基于Qwen微调的实体消歧模型构建的实体链接系统。WDEL通过构建实体链接pipeline，实现了将mention链接到[Wikidata](https://www.wikidata.org)的约1亿个(93,118,252)实体项上的系统。
> [!TIP]
> WDEL主要支持汉语文档中的mention链接到多语言的知识库（Wikidata），但是对于其他语言，同样可以通过微调实现相同的功能。

## 环境依赖
WDEL依赖的环境包括：
* Qwen-7B：[Huggingface :hugs:](https://huggingface.co/Qwen/Qwen-7B-Chat)
* bert-base-multilingual-uncased：[Huggingface :hugs:](https://huggingface.co/google-bert/bert-base-multilingual-uncased)
* MySQL
* requirements.txt中的python package

在终端运行以下命令：
```bash
pip install -r requirements.txt
```

## 运行
1. 准备数据
   1. Wikidata知识库：`KB/wikidata-2023-12-22/tmp/wk_brief_info`，通过`src/wikidata_process/mysql_build.py`导入到MySQL数据库。
   2. 经过候选实体双编码器预先编码的知识库实体向量，通过faiss构建的HNSW索引：`output/cg/train_retriever_be@01-27-12:40/hnsw32_efC512_index.faiss`，以及索引和QID的映射文件：`output/cg/train_retriever_be@01-27-12:40/wk_info_all_pool_idx2qid.pkl`
2. 准备模型参数检查点
   1. 候选生成双编码器，需要`bert-base-multilingual-uncased`，训练好的检查点`output/cg/train_retriever_be@01-27-12:40/pytorch_model.bin`
   2. 实体消歧模型，需要Qwen-7B-Chat的完整参数，以及经过LoRA微调的适配器参数`output/lora/train_lora_reranker@02-11-13:50`
3. 运行实体链接服务
    ```bash
    gunicorn src.pipeline:app -b 0.0.0.0:5000 -t 3600
    # 或
    bash script/run_pipeline.sh
    ```
## 使用示例
上述命令将启动一个简单的 Flask Web 服务器，监听 http://localhost:5000 。你可以使用 POST 请求来调用 /el 端点，传递 doc 和 mentions 参数以获取查询结果。
需要进行实体消歧的文本如下：

>晨报讯【英超】最后一轮【曼联】主场对阵【查尔顿】，【曼联】的锋线搭档是【萨哈】和小将【罗西】，【范尼】再次无缘首发！随后媒体报道称，【荷兰人】在得知自己没有首发后愤怒地提前回家，是什么原因让【弗格森】铁心废掉【老特拉福德】最高效的射手？英伦媒体昨天给出答案———【范尼】是因为跟队友【C罗纳尔多】起了冲突，才被【弗格森】命令离队。“最近这一个星期发生了几次事故，我认为这威胁到了俱乐部的团队精神。在如此重要的一个比赛日，我觉得【范尼】应该离开。”【弗格森】在上周日比赛后的话意味着风暴即将来临，【范尼】做了什么？《【太阳报】》昨天报道称，【范尼】在训练场上与【小小罗】大打出手，而这已经是两人本赛季第二次发生冲突！从【联赛杯】决赛开始就失去主力位置的【范尼】尽管一度保持了沉默，但他显然无法接受赛季最后一轮连大名单都不进的事实，【荷兰人】可能就此离开【曼联】！事情发生后，【范尼】的经纪人说：“【路德】没有离开【曼彻斯特】，他只是离开球队。我们已经决定保持沉默。因为我们不愿把身价降低到【弗格森】那个档次，谁都知道他跟媒体说了什么。”但事实上从【弗格森】对待【贝克汉姆】和【斯塔姆】的事例我们不难发现，下赛季【范尼】再也不可能出现在【老特拉福德】了。目前对【范尼】感兴趣的俱乐部已经不在少数，【AC米兰】、【罗马】甚至【皇马】都渴望得到这位“【禁区之王】”！'


```python
# url 是中文wikipedia的title，仅用于参考，实际使用时不需要提供
query = {
    'doc': '晨报讯 英超最后一轮曼联主场对阵查尔顿，曼联的锋线搭档是萨哈和小将罗西，范尼再次无缘首发！随后媒体报道称，荷兰人在得知自己没有首发后愤怒地提前回家，是什么原因让弗格森铁心废掉老特拉福德最高效的射手？英伦媒体昨天给出答案———范尼是因为跟队友C罗纳尔多起了冲突，才被弗格森命令离队。“最近这一个星期发生了几次事故，我认为这威胁到了俱乐部的团队精神。在如此重要的一个比赛日，我觉得范尼应该离开。”弗格森在上周日比赛后的话意味着风暴即将来临，范尼做了什么？《太阳报》昨天报道称，范尼在训练场上与小小罗大打出手，而这已经是两人本赛季第二次发生冲突！从联赛杯决赛开始就失去主力位置的范尼尽管一度保持了沉默，但他显然无法接受赛季最后一轮连大名单都不进的事实，荷兰人可能就此离开曼联！事情发生后，范尼的经纪人说：“路德没有离开曼彻斯特，他只是离开球队。我们已经决定保持沉默。因为我们不愿把身价降低到弗格森那个档次，谁都知道他跟媒体说了什么。”但事实上从弗格森对待贝克汉姆和斯塔姆的事例我们不难发现，下赛季范尼再也不可能出现在老特拉福德了。目前对范尼感兴趣的俱乐部已经不在少数，AC米兰、罗马甚至皇马都渴望得到这位“禁区之王”！',
    'mentions': [
        {'mention': '英超', 'start': 4, 'end': 6, 'url': '/wiki/英格兰足球超级联赛'},
        {'mention': '曼联', 'start': 10, 'end': 12, 'url': '/wiki/曼彻斯特联足球俱乐部'},
        {'mention': '查尔顿', 'start': 16, 'end': 19, 'url': '/wiki/查尔顿竞技足球俱乐部'},
        {'mention': '曼联', 'start': 20, 'end': 22, 'url': '/wiki/曼彻斯特联足球俱乐部'},
        {'mention': '萨哈', 'start': 28, 'end': 30, 'url': '/wiki/路易·萨哈'},
        {'mention': '罗西', 'start': 33, 'end': 35, 'url': '/wiki/朱塞佩·罗西'},
        {'mention': '范尼', 'start': 36, 'end': 38, 'url': '/wiki/路德·范尼斯特鲁伊'},
        {'mention': '荷兰人', 'start': 53, 'end': 56, 'url': '/wiki/路德·范尼斯特鲁伊'},
        {'mention': '弗格森', 'start': 80, 'end': 83, 'url': '/wiki/亚历克斯·弗格森'},
        {'mention': '老特拉福德', 'start': 87, 'end': 92, 'url': '/wiki/老特拉福德球场'},
        {'mention': '范尼', 'start': 112, 'end': 114, 'url': '/wiki/路德·范尼斯特鲁伊'},
        {'mention': 'C罗纳尔多', 'start': 120, 'end': 125, 'url': '/wiki/基斯坦奴·朗拿度'}
        ...
    ]
}

import requests
import json

# 定义请求数据
data = query

# 发送 POST 请求
url = "http://localhost:5000/el"
headers = {"Content-Type": "application/json"}
response = requests.post(url, data=json.dumps(data), headers=headers)

# 处理响应
if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print(f"Error: {response.status_code}, {response.text}")
```
输出如下所示
```json
[
  [
    {
    "desc": {"en": "English men's association football top league", "zh": "英格兰最高等级的足球赛事"},
    "label": {"en": "Premier League", "zh": "英格兰足球超级联赛"},
    "qid": "Q9448",
    "score": 0.14005205035209656
    },
    {
    "desc": {"en": "17th season of the Premier League", "zh": "足球赛事"},
    "label": {"en": "2008–09 Premier League", "zh": "2008–09赛季英格兰超级联赛"},
    "qid": "Q186869",
    "score": 0.1292843371629715
    }
  ],
  [
    {
    "desc": {"en": "association football club in Manchester, England", "zh": "足球俱乐部"},
    "label": {"en": "Manchester United F.C.", "zh": "曼彻斯特联足球俱乐部"},
    "qid": "Q18656",
    "score": 0.1715932935476303
    },
    {
    "desc": {"en": "association football club in Manchester, England", "zh": null},
    "label": {"en": "F.C. United of Manchester", "zh": "曼市联足球会"},
    "qid": "Q18274",
    "score": 0.13240531086921692
    }
  ],
  
  ...

  [
    {
      "desc": {"en": "French association football player", "zh": null},
      "label": {"en": "Louis Saha", "zh": "路易·萨哈"},
      "qid": "Q484968",
      "score": 0.20607253909111023
    },
    {
      "desc": {"en": "Turkish footballer", "zh": null},
      "label": {"en": "Hasan Şaş", "zh": "哈珊"},
      "qid": "Q313474",
      "score": 0.14765742421150208
    }
 ]
]
```
## 性能
* 系统占用内存约76G，空载时占用显存约18G，推理时占用显存约34G（infer batch size为10），降低函数 `entity_disambiguation` 的batch size参数可以进一步降低显存占用。
* 在Hansel两个测试集上的准确率如下：

| 方法            | 指标 |      KB      | mention | 实体数量 | Hansel FS | Hasnel ZS |
| --------------- | :--: | :----------: | :------: | :------: | :-------: | :-------: |
| WDEL-cg (Exact) | R@10 |   Wikidata   |   中文   |   93M    |   58.5    |   85.9    |
| WDEL-cg (Faiss) | R@10 |   Wikidata   |   中文   |   93M    |   55.8    |   80.1    |
| WDEL            | Acc  |   Wikidata   |   中文   |   93M    |   43.4    |   79.0    |
| mGENRE          | Acc  |   Wikidata   |  多语言  |   20M    |   36.6    |   68.4    |
| CA              | Acc  | Wikipedia zh |   中文   |    1M    |   46.2    |   76.6    |

> [!NOTE]
> * WDEL-cg Faiss: 使用 HNSW 进行近似向量搜索实现候选实体生成，每个顶点的连接数为32；索引构建过程中探索的层深度为512；搜索过程中探索的图层深度为1024。
> * WDEL: 首先基于 WDEL-cg(Exact) 进行精准的最大内积搜索，获取Top-10的候选实体，在此基础上进行实体消歧。
> * mGENRE: 多语言自回归实体链接系统 [[GENRE GitHub repo]](https://github.com/facebookresearch/GENRE) [[mGENRE Paper]](https://arxiv.org/abs/2103.12528)
> * CA: 基于交叉编码器的实体链接 [[Hansel Github repo]](https://github.com/HITsz-TMG/Hansel?tab=readme-ov-file) [[Hansel Paper]](https://dl.acm.org/doi/abs/10.1145/3539597.3570418)

        

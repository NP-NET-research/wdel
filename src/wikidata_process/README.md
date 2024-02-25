# Wikidata数据处理

## 1. Wikidata 三元组数据导入
使用[Qlever](https://github.com/ad-freiburg/qlever?tab=readme-ov-file)将Wikidata数据的三元组数据导入到本地数据库，并进行查询。Qlever是一个SPARQL查询引擎，可以直接通过Web界面或命令行进行查询。Qlever的使用方法可以参考qlever-control的文档[qlever-control](https://github.com/ad-freiburg/qlever-control)，通过Qleverfile实现对Wikidata数据的导入和配置，本项目使用的Qleverfile如下：
```
# Qleverfile for Wikidata, use with https://github.com/ad-freiburg/qlever-control
#
# qlever get-data    downloads two .bz2 files of total size ~100 GB
# qlever index       takes ~7 hours and ~40 GB RAM (on an AMD Ryzen 9 5900X)
# qlever start       starts the server (takes around 30 seconds)

[data]
NAME              = wikidata
GET_DATA_URL      = https://dumps.wikimedia.org/wikidatawiki/entities
GET_DATA_CMD      = curl -LO -C - ${GET_DATA_URL}/latest-truthy.nt.bz2 ${GET_DATA_URL}/latest-lexemes.nt.bz2
INDEX_DESCRIPTION = "Full Wikidata dump from ${GET_DATA_URL} (latest-truthy.nt.bz2 and latest-lexemes.nt.bz2)"

[index]
FILE_NAMES      = wikidata-20231222-lexemes.nt.bz2 wikidata-20231222-truthy.nt.bz2 
CAT_FILES       = bzcat ${FILE_NAMES}
SETTINGS_JSON   = { "languages-internal": ["en"], "prefixes-external": [ "<http://www.wikidata.org/entity/statement", "<http://www.wikidata.org/value", "<http://www.wikidata.org/reference" ], "locale": { "language": "en", "country": "US", "ignore-punctuation": true }, "ascii-prefixes-only": false, "num-triples-per-batch": 10000000 }
WITH_TEXT_INDEX = false
STXXL_MEMORY    = 10g

[server]
PORT                  = 7001
ACCESS_TOKEN          = ${data:NAME}_372483264
MEMORY_FOR_QUERIES    = 100G
CACHE_MAX_SIZE        = 100G

[docker]
USE_DOCKER = true
IMAGE      = adfreiburg/qlever

[ui]
PORT   = 7000
CONFIG = wikidata
```
其中需要下载最新的Wikidata dump: [latest-truthy.nt.bz2](https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.bz2) 和 [latest-lexemes.nt.bz2](https://dumps.wikimedia.org/wikidatawiki/entities/latest-lexemes.nt.bz2)，转储格式为三元组。2023年12月的Wikidata转储中包含约14.78B个三元组，导入Qlever（构建索引）大约需要17h，占用硬盘空间约400GB。

Qlever 运行方法：
```bash
# cd 到目标目录
cd target_folder
# 创建并保存Qleverfile
touch Qleverfile
# 下载数据, 也可以使用 python src/wikidata_process/qlever-control/qlever get-data
wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.bz2
wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-lexemes.nt.bz2
# 重命名文件
mv latest-truthy.nt.bz2 wikidata-20231222-truthy.nt.bz2 
mv latest-lexemes.nt.bz2 wikidata-20231222-lexemes.nt.bz2
# 数据导入qlever
python $repo_path/third_party/qlever-control/qlever index
# 启动qlever
python $repo_path/third_party/qlever-control/qlever restart
# 启动SPARQL Web页面
python $repo_path/third_party/qlever-control/qlever ui
```

## 2. Wikidata SPARQL查询
本项目主要通过命令行进行Wikidata数据查询，命令模板如下：
```bash
curl -s http://127.0.0.1:7001 
    -H "Accept: text/tab-separated-values" \
    -H "Content-type: application/sparql-query" \
    --data "$sparql_query_command_line" > "$save_fpath.tsv"
```

为了避免查询的返回集合占用内存过大，我们分别进行如下查询：
```sparql
PREFIX wikibase: <http://wikiba.se/ontology#> 
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
SELECT ?p ?en_label ?en_desc ?zh_label ?zh_desc
WHERE {
     ?p rdfs:label ?en_label . 
	 ?p a wikibase:Property . 
     FILTER(LANG(?en_label)="en") .
	 OPTIONAL {
	    ?p schema:description ?en_desc . 
		FILTER (LANG(?en_desc) = "en") .
	}
	OPTIONAL {
	    ?p rdfs:label ?zh_label . 
		FILTER (LANG(?zh_label) = "zh") .
	}
	OPTIONAL {
	    ?p schema:description ?zh_desc . 
		FILTER (LANG(?zh_desc) = "zh") .
	}
}
```

```sparql
PREFIX wikibase: <http://wikiba.se/ontology#> 
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?entity ?entity_p31_1 ?entity_p31_2 ?entity_p31_3 
WHERE { 
	?entity a wikibase:Item .
	?entity wdt:P31 ?entity_p31_1 .
	?entity_p31_1 a wikibase:Item .
	OPTIONAL {
		?entity_p31_1 wdt:P31 ?entity_p31_2 .
		?entity_p31_2 a wikibase:Item .
		OPTIONAL {
			?entity_p31_2 wdt:P31 ?entity_p31_3 .
			?entity_p31_3 a wikibase:Item .
		}
	}
}

PREFIX wikibase: <http://wikiba.se/ontology#> 
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?entity ?entity_p279_1 ?entity_p279_2 ?entity_p279_3 
WHERE { 
	?entity a wikibase:Item .
	?entity wdt:P279 ?entity_p279_1 .
	?entity_p279_1 a wikibase:Item .
	OPTIONAL {
		?entity_p279_1 wdt:P279 ?entity_p279_2 .
		?entity_p279_2 a wikibase:Item .
		OPTIONAL {
			?entity_p279_2 wdt:P279 ?entity_p279_3 .
			?entity_p279_3 a wikibase:Item .
		}
	}
}
```
通过下面的SPARQL语句查询出 Wikidata 的 item 之间的重定向关系：
```sparql
PREFIX wikibase: <http://wikiba.se/ontology#> 
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
SELECT ?sub ?obj 
WHERE { 
     ?sub owl:sameAs ?obj .
     ?obj a wikibase:Item . 
}
```

## 3. 处理查询结果
* 包括删除URI前缀：`http://www.wikidata.org/entity/`、繁简转换、删除Wikimedia帮助页面等，主要对应于 `src/wikidata_process/process_string.py` 脚本
* 对不同的数据特征（字段）进行整合，构建成jsonl文件，主要对应于`src/wikidata_process/merge_info.py` 脚本
处理结果示例如下：
```json
{
    "qid": "Q1",
    "label": { "en": "Universe", "zh": "宇宙"},
    "desc": {"en": "all of the spacetime and its contents including the Earth, possibly being part of a multiverse, distinct from parallel universes if they exist", "zh": "行星、恆星、星系、所有物质和能量的总体"},
    "alt": {"en": ["heaven and earth", "yin and yang",  "world", ...], "zh": ["干坤"]},
    "P31": {
        "qid": [["Q36906466", "Q5127848", "Q19478619"], ["Q36906466", "Q5127848", "Q33104279"]],
        "en": [["universe", "class", "metaclass"], ["universe", "class", "philosophical concept"]],
        "zh": [["宇宙", "类", "元类"], ["宇宙", "类", "哲学概念"]]
    },
    "P279": {"qid": [], "en": [], "zh": []}
}
```
* 此外，还将清洗整合后的数据导入了MySQL数据库，方便后续查询使用，构建了三个数据表：Entity、P31、P279；构建和导入数据库的脚本对应于`src/wikidata_process/mysql_build.py`

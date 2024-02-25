import json
import jsonlines
from tqdm import tqdm
import os
import mysql.connector

# MySQL数据库连接信息
db_config = {
    "host": "127.0.0.1",
    "user": os.environ.get('DB_USER'),
    "password": os.environ.get('DB_PASSWORD'),
    # 'database': 'wikidata'
}


def create_table(cursor, database):
    # 切换到指定的数据库
    cursor.execute(f"USE {database}")
    # 创建 Entity 表
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS Entity (
            id INT AUTO_INCREMENT PRIMARY KEY,
            qid TEXT NOT NULL,
            label_en TEXT,
            label_zh TEXT,
            desc_en TEXT,
            desc_zh TEXT,
            alt_en TEXT,
            alt_zh TEXT
        )
    """
    )

    # 创建 P31 表
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS P31 (
            id INT AUTO_INCREMENT PRIMARY KEY,
            entity_id INT,
            qid TEXT,
            p31_qid TEXT,
            p31_en TEXT,
            p31_zh TEXT,
            FOREIGN KEY (entity_id) REFERENCES Entity(id)
        )
    """
    )

    # 创建 P279 表
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS P279 (
            id INT AUTO_INCREMENT PRIMARY KEY,
            entity_id INT,
            qid TEXT,
            p279_qid TEXT,
            p279_en TEXT,
            p279_zh TEXT,
            FOREIGN KEY (entity_id) REFERENCES Entity(id)
        )
    """
    )


def connect_mysql(database):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute(f"SHOW DATABASES LIKE '{database}'")
        result = cursor.fetchone()
        if result is None:
            # 数据库不存在，创建数据库
            print(f"Database {database} does not exist, creating...")
            # 创建数据库
            cursor.execute(f"CREATE DATABASE {database}")
            create_table(cursor, database)
            print(f"Database {database} created successfully!")

        else:
            print(f"Database {database} already exists!")

        print("MySQL connected successfully!")

        return connection
    except mysql.connector.Error as error:
        print(f"Failed to connect to MySQL: {error}")
        return None


# 读取JSONL文件，并将数据插入数据库
def import_data(file_path, connection, cursor):
    try:
        with jsonlines.open(file_path, "r") as file:
            for line in file:
                insert_entity(cursor, line)
        connection.commit()
        print(f"{file_path} import successful!")

    except Exception as e:
        print(f"Error: {e}")
        

# 插入实体数据
def insert_entity(cursor, data):
    # 插入 Entity 表数据
    cursor.execute(
        """
        INSERT INTO Entity (qid, label_en, label_zh, desc_en, desc_zh, alt_en, alt_zh)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        label_en = VALUES(label_en), label_zh = VALUES(label_zh),
        desc_en = VALUES(desc_en), desc_zh = VALUES(desc_zh),
        alt_en = VALUES(alt_en), alt_zh = VALUES(alt_zh)
    """,
        (
            data["qid"],
            data["label"]["en"],
            data["label"]["zh"],
            data["desc"]["en"],
            data["desc"]["zh"],
            json.dumps(data["alt"]["en"], ensure_ascii=False),
            json.dumps(data["alt"]["zh"], ensure_ascii=False),
        ),
    )

    # 获取插入的 Entity 表的 ID
    entity_id = cursor.lastrowid

    # 插入 P31 表数据
    insert_relation(cursor, entity_id, "P31", data["qid"], data["P31"])

    # 插入 P279 表数据
    insert_relation(cursor, entity_id, "P279", data["qid"], data["P279"])


# 插入关系数据
def insert_relation(cursor, entity_id, relation_type, qid, relation_data):
    for i in range(len(relation_data["qid"])):
        cursor.execute(
            f"""
            INSERT INTO {relation_type} (entity_id, qid, {relation_type.lower()}_qid, {relation_type.lower()}_en, {relation_type.lower()}_zh)
            VALUES (%s, %s, %s, %s, %s)
        """,
            (
                entity_id,
                qid,
                json.dumps(relation_data["qid"][i], ensure_ascii=False),
                json.dumps(relation_data["en"][i], ensure_ascii=False),
                json.dumps(relation_data["zh"][i], ensure_ascii=False),
            ),
        )


# 从文件夹中导入数据
def load_kb_from_folder(dir):
    database = "wikidata"
    connection = connect_mysql(database)
    cursor = connection.cursor()
    cursor.execute(f"USE {database}")

    split_index_list = sorted([int(file.lstrip("wk_info_").rstrip(".jsonl")) for file in os.listdir(dir)])[1:]
    for split_index in tqdm(split_index_list, dynamic_ncols=True):
        fpath = os.path.join(dir, f"wk_info_{split_index}.jsonl")
        import_data(fpath, connection=connection, cursor=cursor)

    if connection and connection.is_connected():
        cursor.close()
        connection.close()
        print("Connection closed.")


def create_index(database):
    try:
        connection = connect_mysql(database)
        cursor = connection.cursor()
        # 切换到指定的数据库
        cursor.execute(f"USE {database}")
        # 创建索引
        cursor.execute("CREATE INDEX idx_Entity_qid ON Entity (qid(15))")
        cursor.execute("CREATE INDEX idx_P31_entity_id ON P31 (entity_id)")
        cursor.execute("CREATE INDEX idx_P279_entity_id ON P279 (entity_id)")
        # 提交事务
        connection.commit()
        print("Index created successfully!")
    except mysql.connector.Error as error:
        print(f"Failed to create index: {error}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            print("Connection closed.")


if __name__ == "__main__":
    # kb_folder = "KB/wikidata-2023-12-22/tmp/wk_brief_info"

    # load_kb_from_folder(kb_folder)
    create_index("wikidata")

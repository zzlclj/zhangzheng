# 必须放在所有 import 最开头
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
# 下面原有导入后续再优化
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 原有全部代码不变
csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'campus_data.csv')
df = pd.read_csv(csv_path)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh",
    model_kwargs={"trust_remote_code": True}
)
# 后续构建向量库代码...

# 准备文本和元数据（以回答内容为文本，元数据包含id, category, question, source）
texts = df['answer'].tolist()
metadatas = df[['id', 'category', 'question', 'source']].to_dict('records')


# 创建向量库并持久化
persist_dir = os.path.join(os.path.dirname(__file__), '..', 'vector_db')
vector_db = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,          # 改为 metadatas
    persist_directory=persist_dir
)
# 新版本 Chroma 会自动持久化，无需显式调用 .persist()
# 但为了兼容，可以调用（如果存在）
try:
    vector_db.persist()
except AttributeError:
    pass

print(f"✅ 向量库构建完成，共存入 {len(texts)} 条记录")
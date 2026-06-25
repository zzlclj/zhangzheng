import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import re
import requests
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from prompt_templates import RAG_PROMPT
from tools import get_current_week, calculate_gpa

load_dotenv()

# ------------------- 缓存资源 -------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"trust_remote_code": True}
    )

@st.cache_resource
def load_vector_db():
    embeddings = load_embeddings()
    return Chroma(persist_directory="./vector_db", embedding_function=embeddings)

embeddings = load_embeddings()
vector_db = load_vector_db()

# ------------------- Spark X API 配置 -------------------
SPARK_APIPASSWORD = os.getenv("SPARK_APIPASSWORD")
SPARK_HTTP_URL = os.getenv("SPARK_HTTP_URL", "https://spark-api-open.xf-yun.com/x2/chat/completions")
SPARK_MODEL = os.getenv("SPARK_MODEL", "spark-x")

if not SPARK_APIPASSWORD:
    st.error("❌ 请在 .env 文件中设置 SPARK_APIPASSWORD")
    st.stop()

# ------------------- 调用 Spark X HTTP API -------------------
def call_spark_api(user_message):
    """调用讯飞星火 Spark X HTTP API"""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SPARK_APIPASSWORD}"
    }
    
    payload = {
        "model": SPARK_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个校园生活助手，请用中文回答，回答要清晰、简洁、有帮助。"},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.3,
        "max_tokens": 2048
    }
    
    try:
        print(f"请求URL: {SPARK_HTTP_URL}")
        print(f"请求体: {json.dumps(payload, ensure_ascii=False)[:200]}...")
        
        resp = requests.post(SPARK_HTTP_URL, headers=headers, json=payload, timeout=60)
        
        print(f"响应状态码: {resp.status_code}")
        print(f"响应内容: {resp.text[:300]}")
        
        if resp.status_code == 200:
            result = resp.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return f"⚠️ API 返回格式异常: {result}"
        else:
            return f"❌ API 错误：{resp.status_code}\n{resp.text}"
    except requests.exceptions.Timeout:
        return "⚠️ 请求超时，请稍后重试"
    except Exception as e:
        return f"⚠️ 请求异常：{e}"

# ------------------- RAG 问答 -------------------
def rag_retrieve_answer(question):
    docs = vector_db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    prompt_text = RAG_PROMPT.format(context=context, question=question)
    return call_spark_api(prompt_text)

# ------------------- 智能体路由 -------------------
def agent_answer(question):
    if re.search(r'第.*周|校历|本周|几周', question):
        return get_current_week()
    if re.search(r'绩点|GPA|平均分|分数', question):
        nums = re.findall(r'\d+', question)
        if nums:
            return calculate_gpa(','.join(nums))
        else:
            return "请提供您的各科分数，例如：85,90,78"
    return rag_retrieve_answer(question)

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="校园百事通", page_icon="🏫")
st.title("🏫 校园生活百事通助手")
st.markdown("我可以回答校园问题，还能查询校历周数和计算绩点！")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入你的校园问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            answer = agent_answer(prompt)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
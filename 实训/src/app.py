import os
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
import streamlit as st
import re
import requests
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from prompt_templates import RAG_PROMPT
from tools import get_current_week, calculate_gpa
from string import Template

load_dotenv()

# ================== 全局资源初始化 ==================
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

SPARK_APIPASSWORD = os.getenv("SPARK_APIPASSWORD")
SPARK_HTTP_URL = os.getenv("SPARK_HTTP_URL", "https://spark-api-open.xf-yun.com/x2/chat/completions")
SPARK_MODEL = os.getenv("SPARK_MODEL", "spark-x")

# ================== 函数定义 ==================
def call_spark_api(messages):
    if not SPARK_APIPASSWORD:
        return "❌ 未配置 SPARK_APIPASSWORD，请在 .env 中设置"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SPARK_APIPASSWORD}"
    }
    payload = {
        "model": SPARK_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 2048
    }
    try:
        resp = requests.post(SPARK_HTTP_URL, headers=headers, json=payload, timeout=60)
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

def rag_retrieve_answer(question, history):
    try:
        docs = vector_db.similarity_search(question, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        system_prompt = RAG_PROMPT.format(context=context, question=question)
        recent_history = history[-4:] if history else []
        messages = [{"role": "system", "content": system_prompt}] + recent_history + [{"role": "user", "content": question}]
        return call_spark_api(messages)
    except Exception as e:
        return f"⚠️ RAG 检索失败: {str(e)}"

def agent_answer(question, history):
    try:
        if re.search(r'第.*周|校历|本周|几周', question):
            return get_current_week()
        if re.search(r'绩点|GPA|平均分|分数', question):
            nums = re.findall(r'\d+', question)
            if nums:
                return calculate_gpa(','.join(nums))
            else:
                return "请提供您的各科分数，例如：85,90,78"
        return rag_retrieve_answer(question, history)
    except Exception as e:
        return f"⚠️ 处理请求时出错: {str(e)}"

def process_user_input(prompt):
    if not prompt or not prompt.strip():
        return
    prompt = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": prompt})
    history = st.session_state.messages[:-1]
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 正在思考中..."):
            answer = agent_answer(prompt, history)
            print(f"DEBUG: 回答内容: {answer}")
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_answer = answer
    st.session_state.answer_version = st.session_state.get("answer_version", 0) + 1
    st.rerun()

# ================== 语音输入处理（页面渲染前） ==================
voice_input_text = st.query_params.get("voice_input", None)
if voice_input_text:
    st.query_params.clear()
    st.session_state.voice_pending = voice_input_text
    st.rerun()
    st.stop()

if "voice_pending" in st.session_state and st.session_state.voice_pending:
    pending_voice = st.session_state.voice_pending
    st.session_state.voice_pending = None
    process_user_input(pending_voice)
    st.stop()

# ================== 页面配置 ==================
st.set_page_config(
    page_title="校园百事通",
    page_icon="🏫",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ================== 语音组件 HTML 模板 ==================
VOICE_HTML_TEMPLATE = """
<div id="voice-container" style="display:flex; align-items:center; gap:10px; margin-top:8px;">
  <button id="mic-btn" style="background: #4a90d9; color: white; border: none; border-radius: 50%; width: 48px; height: 48px; font-size: 24px; cursor: pointer; box-shadow: 0 4px 12px rgba(74,144,217,0.3); transition: all 0.2s; display: flex; align-items: center; justify-content: center;">🎤</button>
  <span id="status-label" style="color:#4a6a85; font-size:0.9rem;">点击授权并说话</span>
</div>
<script>
  (function() {
    // ---------- 语音输出（TTS）----------
    const answerText = $text;
    const answerVersion = $version;
    const ttsEnabled = $tts_enabled === 'true';   // ← 新增开关
    const lastVersion = sessionStorage.getItem('voice_last_version') || '0';
    // 仅在开启 TTS 且版本变化时朗读
    if (ttsEnabled && String(answerVersion) !== String(lastVersion) && answerText) {
      sessionStorage.setItem('voice_last_version', String(answerVersion));
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(answerText);
        utterance.lang = navigator.language || 'zh-CN';
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
      }
    }

    // ---------- 语音输入（同上，保持不变） ----------
    let recognition = null;
    let isListening = false;
    let finalTranscript = '';
    let submitted = false;
    let startTime = 0;
    let retryTimer = null;
    let permissionGranted = false;
    let micStream = null;

    const statusLabel = document.getElementById('status-label');
    const micBtn = document.getElementById('mic-btn');

    function initRecognition() {
      if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        statusLabel.textContent = '⚠️ 浏览器不支持语音识别';
        micBtn.style.opacity = '0.5';
        return false;
      }
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognition = new SpeechRecognition();
      recognition.lang = navigator.language || 'zh-CN';
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;

      recognition.onresult = function(event) {
        let latest = '';
        for (let i = event.results.length - 1; i >= 0; i--) {
          const result = event.results[i];
          if (result.isFinal) {
            latest = result[0].transcript.trim();
            break;
          } else if (!latest) {
            latest = result[0].transcript.trim();
          }
        }
        if (latest) {
          finalTranscript = latest;
          statusLabel.textContent = '🗣️ ' + latest;
        }
      };

      recognition.onend = function() {
        isListening = false;
        micBtn.style.background = '#4a90d9';
        micBtn.style.transform = 'scale(1)';
        if (finalTranscript && !submitted) {
          submitted = true;
          statusLabel.textContent = '✅ 已识别: ' + finalTranscript;
          const currentUrl = window.parent.location.href;
          const separator = currentUrl.includes('?') ? '&' : '?';
          const newUrl = currentUrl + separator + 'voice_input=' + encodeURIComponent(finalTranscript);
          window.parent.location.href = newUrl;
          return;
        }
        if (!finalTranscript && !submitted) {
          statusLabel.textContent = '⏳ 等待识别结果...';
          if (retryTimer) clearTimeout(retryTimer);
          retryTimer = setTimeout(function() {
            retryTimer = null;
            if (finalTranscript && !submitted) {
              submitted = true;
              statusLabel.textContent = '✅ 已识别: ' + finalTranscript;
              const currentUrl = window.parent.location.href;
              const separator = currentUrl.includes('?') ? '&' : '?';
              const newUrl = currentUrl + separator + 'voice_input=' + encodeURIComponent(finalTranscript);
              window.parent.location.href = newUrl;
            } else {
              statusLabel.textContent = '未检测到语音，请按住说话';
            }
          }, 500);
        } else {
          if (!finalTranscript && !submitted) {
            statusLabel.textContent = '未检测到语音，请按住说话';
          }
        }
      };

      recognition.onerror = function(event) {
        console.error('语音识别错误:', event.error);
        isListening = false;
        micBtn.style.background = '#4a90d9';
        micBtn.style.transform = 'scale(1)';
        let msg = '❌ 识别失败: ';
        switch (event.error) {
          case 'not-allowed': msg += '请点击地址栏🔒允许麦克风'; break;
          case 'no-speech': msg += '没有检测到语音'; break;
          case 'audio-capture': msg += '麦克风不可用，请检查设备'; break;
          case 'network': msg += '网络问题，请重试'; break;
          default: msg += event.error;
        }
        statusLabel.textContent = msg;
        finalTranscript = '';
        if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
      };
      return true;
    }

    function requestPermission() {
      return new Promise((resolve, reject) => {
        if (permissionGranted) {
          resolve(true);
          return;
        }
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          reject('浏览器不支持 getUserMedia');
          return;
        }
        navigator.mediaDevices.getUserMedia({ audio: true })
          .then(function(stream) {
            micStream = stream;
            permissionGranted = true;
            statusLabel.textContent = '✅ 麦克风已授权，按住说话';
            if (!recognition) {
              initRecognition();
            }
            resolve(true);
          })
          .catch(function(err) {
            console.error('权限请求失败:', err);
            let msg = '❌ 权限被拒: ';
            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
              msg += '请手动允许麦克风（点击地址栏🔒 → 权限 → 麦克风 → 允许）';
            } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
              msg += '未找到麦克风设备，请插入麦克风';
            } else {
              msg += err.message;
            }
            statusLabel.textContent = msg;
            reject(err);
          });
      });
    }

    function startRecording() {
      if (!permissionGranted) {
        statusLabel.textContent = '⏳ 正在请求麦克风权限...';
        requestPermission()
          .then(function() {
            startRecording();
          })
          .catch(function(err) {});
        return;
      }
      if (!recognition) {
        if (!initRecognition()) return;
      }
      if (isListening) {
        try { recognition.stop(); } catch(e) {}
        isListening = false;
      }
      finalTranscript = '';
      submitted = false;
      startTime = Date.now();
      try {
        recognition.start();
        isListening = true;
        micBtn.style.background = '#e74c3c';
        micBtn.style.transform = 'scale(1.1)';
        statusLabel.textContent = '🎙️ 聆听中... 松手发送';
      } catch (e) {
        console.error('启动语音识别失败:', e);
        statusLabel.textContent = '⚠️ 启动失败，请重试';
        isListening = false;
      }
    }

    function stopAndSubmit() {
      if (!isListening) return;
      if (Date.now() - startTime < 300) {
        try { recognition.stop(); } catch(e) {}
        isListening = false;
        micBtn.style.background = '#4a90d9';
        micBtn.style.transform = 'scale(1)';
        statusLabel.textContent = '按住说话，松手发送';
        return;
      }
      try {
        recognition.stop();
      } catch(e) {
        console.error('停止识别失败:', e);
      }
    }

    micBtn.addEventListener('mousedown', function(e) {
      e.preventDefault();
      startRecording();
    });
    micBtn.addEventListener('mouseup', function(e) {
      e.preventDefault();
      stopAndSubmit();
    });
    micBtn.addEventListener('touchstart', function(e) {
      e.preventDefault();
      startRecording();
    });
    micBtn.addEventListener('touchend', function(e) {
      e.preventDefault();
      stopAndSubmit();
    });
    micBtn.addEventListener('touchcancel', function(e) {
      if (isListening) stopAndSubmit();
    });

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      statusLabel.textContent = '点击按钮授权麦克风';
    } else {
      statusLabel.textContent = '⚠️ 浏览器不支持麦克风';
      micBtn.style.opacity = '0.5';
    }
  })();
</script>
"""

# ================== 自定义 CSS（保持不变） ==================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(145deg, #f0f4f8 0%, #dae5f0 100%);
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #1a2a3a;
        text-align: center;
        margin-bottom: 0.1rem;
        letter-spacing: 1px;
    }
    .sub-title {
        text-align: center;
        color: #4a6a85;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
        letter-spacing: 2px;
    }
    .sub-title span {
        background: rgba(74, 144, 217, 0.12);
        padding: 0.2rem 1rem;
        border-radius: 20px;
    }
    .quick-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 12px;
        margin: 0.5rem 0 1.2rem 0;
    }
    .quick-btn {
        background: white;
        border: none;
        border-radius: 16px;
        padding: 14px 12px;
        text-align: center;
        font-size: 0.92rem;
        font-weight: 500;
        color: #1e293b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: all 0.25s ease;
        cursor: pointer;
        border: 1px solid #e8edf3;
        line-height: 1.5;
    }
    .quick-btn:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(74, 144, 217, 0.18);
        border-color: #4a90d9;
        background: #f8faff;
    }
    .quick-btn .icon {
        font-size: 1.5rem;
        display: block;
        margin-bottom: 4px;
    }
    .quick-btn .label {
        font-size: 0.85rem;
    }
    .welcome-card {
        background: rgba(255,255,255,0.70);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        padding: 1.8rem 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.5);
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        margin: 0.5rem 0 1rem 0;
    }
    .welcome-card h3 {
        color: #1a2a3a;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }
    .welcome-card p {
        color: #5a7a95;
        font-size: 0.95rem;
        margin: 0;
    }
    .stChatMessage {
        border-radius: 18px !important;
        padding: 12px 18px !important;
        margin: 6px 0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: none !important;
    }
    div[data-testid="stChatMessage"][data-role="user"] {
        background: #4a90d9 !important;
        color: white !important;
        border-top-right-radius: 4px !important;
    }
    div[data-testid="stChatMessage"][data-role="assistant"] {
        background: white !important;
        color: #1e293b !important;
        border-top-left-radius: 4px !important;
        border: 1px solid #e2e8f0 !important;
    }
    .stChatInput > div > div > textarea {
        border-radius: 30px !important;
        padding: 10px 20px !important;
        border: 2px solid #d1d9e6 !important;
        font-size: 1rem !important;
        transition: all 0.3s;
        background: white !important;
    }
    .stChatInput > div > div > textarea:focus {
        border-color: #4a90d9 !important;
        box-shadow: 0 0 0 4px rgba(74,144,217,0.15) !important;
    }
    .css-1d391kg {
        background: rgba(255,255,255,0.92) !important;
        backdrop-filter: blur(6px);
        border-right: 1px solid rgba(0,0,0,0.04);
    }
    .sidebar-content {
        padding: 0.5rem 0.3rem;
    }
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    .feature-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        margin: 6px 0;
        background: #f8fafc;
        border-radius: 12px;
        border-left: 4px solid #4a90d9;
    }
    .feature-icon { font-size: 1.3rem; }
    .feature-text { font-size: 0.92rem; color: #334155; }
    .feature-text b { color: #1a2a3a; }
    .sidebar-footer {
        font-size: 0.8rem;
        color: #94a3b8;
        text-align: center;
        margin-top: 1rem;
        padding-top: 0.8rem;
        border-top: 1px solid #e8edf3;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    @media (max-width: 640px) {
        .quick-grid { grid-template-columns: repeat(2, 1fr); }
        .main-title { font-size: 1.8rem; }
        .welcome-card { padding: 1.2rem; }
    }
</style>
""", unsafe_allow_html=True)

# ================== 侧边栏 ==================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <div class="sidebar-title">🧭 导航</div>
        <div style="margin-bottom:1.2rem;">
            <div class="feature-item">
                <span class="feature-icon">📚</span>
                <span class="feature-text"><b>校园问答</b><br>请假 · 奖学金 · 报修</span>
            </div>
            <div class="feature-item">
                <span class="feature-icon">📅</span>
                <span class="feature-text"><b>校历查询</b><br>当前是第几周</span>
            </div>
            <div class="feature-item">
                <span class="feature-icon">📊</span>
                <span class="feature-text"><b>绩点计算</b><br>输入分数自动算</span>
            </div>
        </div>
        <hr style="margin: 1rem 0; border-color: #e8edf3;">
        <div style="font-size:0.88rem; color:#4a6a85; text-align:center; line-height:1.7;">
            💡 直接输入问题即可<br>
            <span style="font-size:0.82rem; color:#6a8aa5;">“怎么请病假？” · “现在第几周？”</span>
        </div>
        <div style="margin-top: 1rem; padding: 0.8rem 0; border-top: 1px solid #e8edf3;">
            <div style="font-size:0.85rem; color:#4a6a85; margin-bottom:4px;">🎤 语音输入</div>
            <div id="voice-wrapper" style="background: #f8fafc; border-radius: 16px; padding: 6px 12px;">
            </div>
        </div>
        <!-- ===== 新增：语音播报开关 ===== -->
        <div style="margin-top: 0.5rem;">
            <label style="font-size:0.85rem; color:#4a6a85; display:flex; align-items:center; gap:6px;">
                <span>🔊 语音播报</span>
            </label>
            <div style="background: #f8fafc; border-radius: 12px; padding: 6px 12px; margin-top: 4px;">
                <span style="font-size:0.85rem; color:#4a6a85;">
                <input type="checkbox" id="tts-toggle" style="transform: scale(1.2); margin-right: 8px;" 
                       onchange="this.checked ? setTTS(true) : setTTS(false)">
                播报回答
                </span>
                <script>
                function setTTS(enabled) {
                    const currentUrl = window.parent.location.href;
                    const sep = currentUrl.includes('?') ? '&' : '?';
                    window.parent.location.href = currentUrl + sep + 'tts=' + (enabled ? 'true' : 'false');
                }
                // 读取当前状态并设置复选框
                const params = new URLSearchParams(window.parent.location.search);
                const ttsParam = params.get('tts');
                if (ttsParam !== null) {
                    const cb = document.getElementById('tts-toggle');
                    if (cb) cb.checked = (ttsParam === 'true');
                }
                </script>
            </div>
        </div>
        <div class="sidebar-footer">
            🏫 安徽交通职业技术学院<br>
            RAG 智能助手 v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== 渲染语音组件 =====
    # 处理 tts 参数（来自复选框）
    tts_param = st.query_params.get("tts", None)
    if tts_param is not None:
        st.session_state.tts_enabled = (tts_param == "true")
        # 清除参数防止重复
        st.query_params.clear()
        # 但不清除 voice_input，避免干扰；只清除 tts
        # 我们保留其他参数，但此时只关心 tts
        # 重新运行以应用新状态
        st.rerun()

    # 确保 tts_enabled 存在
    if "tts_enabled" not in st.session_state:
        st.session_state.tts_enabled = True

    last_answer = st.session_state.get("last_answer", "")
    answer_version = st.session_state.get("answer_version", 0)
    answer_text_json = json.dumps(last_answer)
    template = Template(VOICE_HTML_TEMPLATE)
    voice_html = template.substitute(
        text=answer_text_json,
        version=answer_version,
        tts_enabled=str(st.session_state.tts_enabled).lower()
    )
    st.components.v1.html(voice_html, height=100)

# ================== 主界面 ==================
st.markdown("""
<div class="main-title">🏫 校园生活百事通</div>
<div class="sub-title"><span>✨ 智能问答 · 校历查询 · 绩点计算</span></div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
quick_questions = [
    ("📋", "怎么请病假？"),
    ("🏆", "奖学金要多少绩点？"),
    ("🔧", "宿舍怎么报修？"),
    ("💳", "一卡通丢了怎么办？"),
]
with col1:
    if st.button("📋\n怎么请病假？", key="q1", use_container_width=True):
        st.session_state.quick_prompt = quick_questions[0][1]
with col2:
    if st.button("🏆\n奖学金要多少绩点？", key="q2", use_container_width=True):
        st.session_state.quick_prompt = quick_questions[1][1]
with col3:
    if st.button("🔧\n宿舍怎么报修？", key="q3", use_container_width=True):
        st.session_state.quick_prompt = quick_questions[2][1]
with col4:
    if st.button("💳\n一卡通丢了怎么办？", key="q4", use_container_width=True):
        st.session_state.quick_prompt = quick_questions[3][1]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "answer_version" not in st.session_state:
    st.session_state.answer_version = 0

for msg in st.session_state.messages:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-card">
        <h3>👋 你好！我是你的校园百事通</h3>
        <p>我可以帮你解答校园生活问题、查询校历、计算绩点。<br>
        试试在上方点击快捷问题，或直接输入你的问题吧！</p>
    </div>
    """, unsafe_allow_html=True)

pending = st.session_state.pop("quick_prompt", None)
if pending:
    process_user_input(pending)
    st.stop()

if prompt := st.chat_input("💬 输入你的校园问题..."):
    process_user_input(prompt)
    st.stop()

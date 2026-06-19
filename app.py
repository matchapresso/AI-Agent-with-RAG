import streamlit as st
import operator
import time
from typing import Annotated, List, TypedDict, Union

# LangChain & LangGraph Imports
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Streamlit Page Configuration for Language switching and session state management
LOCALES = {
    "id": {
        "page_title": "Qwen 2.5 LangGraph AI Agent",
        "sidebar_header": "⚙️ Konfigurasi Sistem",
        "sidebar_desc": "Sesuaikan basis pengetahuan dan pengaturan AI Agent di sini.",
        "lang_label": "🌐 Pilih Bahasa / Language:",
        "arxiv_label": "Arxiv Paper ID:",
        "btn_init": "🚀 Bangun & Inisialisasi Agent",
        "success_init": "✅ Agent & Knowledge Base Berhasil Diaktifkan!",
        "sidebar_util": "🛠️ Utilitas",
        "btn_clear": "🗑️ Hapus Riwayat Obrolan",
        "title": "🤖 LangGraph AI Agent dengan Hybrid RAG",
        "welcome": "Selamat datang! Sistem ini menggunakan model **Qwen 2.5 (via HuggingFace)** yang diorkestrasi oleh **LangGraph** untuk melakukan penalaran mandiri (*autonomous reasoning*) dan menggunakan *external tools* untuk menyelesaikan instruksi Anda.",
        "info_init": "💡 **Langkah Awal:** Silakan klik tombol **'Bangun & Inisialisasi Agent'** di panel samping untuk memuat database RAG dan mengaktifkan agen.",
        "status_ready": "🟢 AI Agent siap menerima perintah (RAG, Math, Summarization, Text Transformation).",
        "chat_placeholder": "Tanyakan detail paper, hitungan matematika, atau transformasi teks...",
        "status_thinking": "🧠 Agent sedang berpikir & memilih tools...",
        "status_tool": "🛠️ **Mengaktifkan Alat:**",
        "status_arg": "📥 **Argumen:**",
        "status_output": "📤 **Hasil dari Alat:**",
        "status_complete": "✨ Pemrosesan Selesai!",
        "err_db": "Error: Pangkalan data (Knowledge Base) belum siap atau belum dibangun.",
        "err_arxiv": "❌ Gagal memuat dokumen dari Arxiv. Silakan periksa kembali Arxiv ID.",
        "loading_arxiv": "📥 Mengunduh dan memproses paper Arxiv ID:"
    },
    "en": {
        "page_title": "Qwen 2.5 LangGraph AI Agent",
        "sidebar_header": "⚙️ System Configuration",
        "sidebar_desc": "Customize the knowledge base and AI Agent settings here.",
        "lang_label": "🌐 Pilih Bahasa / Language:",
        "arxiv_label": "Arxiv Paper ID:",
        "btn_init": "🚀 Build & Initialize Agent",
        "success_init": "✅ Agent & Knowledge Base Successfully Activated!",
        "sidebar_util": "🛠️ Utilities",
        "btn_clear": "🗑️ Clear Chat History",
        "title": "🤖 LangGraph AI Agent with Hybrid RAG",
        "welcome": "Welcome! This system uses the **Qwen 2.5 (via HuggingFace)** model orchestrated by **LangGraph** to perform autonomous reasoning and use external tools to solve your instructions.",
        "info_init": "💡 **Initial Step:** Please click the **'Build & Initialize Agent'** button in the sidebar to load the RAG database and activate the agent.",
        "status_ready": "🟢 AI Agent is ready to receive commands (RAG, Math, Summarization, Text Transformation).",
        "chat_placeholder": "Ask about paper details, mathematical calculations, or text transformations...",
        "status_thinking": "🧠 Agent is thinking & selecting tools...",
        "status_tool": "🛠️ **Activating Tool:**",
        "status_arg": "📥 **Arguments:**",
        "status_output": "📤 **Tool Output:**",
        "status_complete": "✨ Processing Complete!",
        "err_db": "Error: Knowledge Base is not ready or has not been built yet.",
        "err_arxiv": "❌ Failed to load document from Arxiv. Please check the Arxiv ID again.",
        "loading_arxiv": "📥 Downloading and processing Arxiv paper ID:"
    }
}
# 1. STREAMLIT PAGE CONFIGURATION & SESSION STATE
st.set_page_config(
    page_title="end-to-end LangGraph Agent with Hybrid RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "agent_app" not in st.session_state:
    st.session_state.agent_app = None

if "lang" not in st.session_state:
    st.session_state.lang = "id"  # Default Bahasa Indonesia

# Global LLM instance (deterministic outputs for reliability)
@st.cache_resource
def get_llm_engine():
    # Setup Endpoint API ke HuggingFace
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        max_new_tokens=1024,
        temperature=0.1, # Tetap rendah agar agen tidak berhalusinasi saat memanggil tool
        # Pastikan kamu sudah set HF_TOKEN di Streamlit Secrets atau Environment Variable
        huggingfacehub_api_token=st.secrets["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"] 
    )
    # ChatHuggingFace bertugas memastikan format pesannya kompatibel dengan LangGraph Agent
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model

llm_engine = get_llm_engine()

#language mapping for UI text
t = LOCALES[st.session_state.lang]

# 2. KNOWLEDGE BASE (HYBRID RAG PIPELINE)
def build_knowledge_base(arxiv_id: str):
    """
    Builds a Hybrid Search Index (Vector + Keyword) using Chroma and BM25.
    """
    with st.spinner(f"📥 Mengunduh dan memproses paper Arxiv ID: {arxiv_id}..."):
        # A. Load Data from Arxiv
        loader = ArxivLoader(query=arxiv_id, load_max_docs=1)
        raw_docs = loader.load()
        
        if not raw_docs:
            st.error("❌ Gagal memuat dokumen dari Arxiv. Silakan periksa kembali Arxiv ID.")
            return None

        # B. Split Text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = text_splitter.split_documents(raw_docs)

        # C. Generate Embeddings (GPU Accelerated if available, fallback to CPU)
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5",
                model_kwargs={'device': 'cuda'}
            )
        except Exception:
            # Fallback to CPU if CUDA is not configured in the running environment
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5",
                model_kwargs={'device': 'cpu'}
            )

        # D. Dense Vector Store Index
        vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="agent_rag_db")
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # E. Sparse Keyword Search Index
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 3

        # F. Ensemble Hybrid Retriever (50/50 Weights)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.5, 0.5]
        )
        return ensemble_retriever

# 3. LANGGRAPH AGENT & TOOLS DEFINITIONS
@tool
def rag_search(query: str) -> str:
    """
    Search for technical details about the 'Attention Is All You Need' paper,
    Transformer architecture, or self-attention mechanisms.
    """
    if st.session_state.rag_engine is None:
        return "Error: Pangkalan data (Knowledge Base) belum siap atau belum dibangun."
    
    docs = st.session_state.rag_engine.invoke(query)
    return "\n\n".join([f"[Source: Page {d.metadata.get('page','?')}] {d.page_content}" for d in docs])

@tool
def calculator(expression: str) -> str:
    """
    Calculates mathematical expressions safely.
    Input examples: '200 / 5' or '10 + 5 * 2'.
    """
    try:
        allowed_names = {"abs": abs, "min": min, "max": max, "round": round}
        return str(eval(expression, {"__builtins__": None}, allowed_names))
    except Exception as e:
        return f"Math Error: {e}"

@tool
def summarizer(text: str) -> str:
    """Summarizes a long piece of text into a concise paragraph."""
    return llm_engine.invoke(f"Summarize this text in 2 sentences:\n{text}").content

@tool
def text_transformer(text: str, operation: str) -> str:
    """
    Transforms text strings.
    Valid operations: 'upper' (uppercase), 'lower' (lowercase), 'reverse' (reverses text).
    """
    if operation == "upper": return text.upper()
    elif operation == "lower": return text.lower()
    elif operation == "reverse": return text[::-1]
    return "Error: Operasi tidak dikenal."

def compile_agent_graph():
    """
    Compiles the LangGraph StateGraph (ReAct Agent Workflow).
    """
    tools_list = [rag_search, calculator, summarizer, text_transformer]
    llm_with_tools = llm_engine.bind_tools(tools_list)

    class AgentState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]

    def agent_node(state):
        # Menyisipkan instruksi bahasa respons di System Prompt internal agen
        system_instruction = f" Respond to the user strictly in the language they used or prefer. Current interface language is set to: {st.session_state.lang}."
        messages = state['messages']
        return {"messages": [llm_with_tools.invoke(state['messages'])]}

    def router(state):
        if state['messages'][-1].tool_calls:
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools_list))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", router, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile()

# 4. USER INTERACTION & LAYOUT
# --- SIDEBAR PANEL ---
with st.sidebar:
    st.header(t["sidebar_header"])
    st.markdown(t["sidebar_desc"])
    
    # 🌐 FITUR UTAMA: Switch Bahasa
    lang_options = ["Bahasa Indonesia", "English"]
    current_lang_idx = 0 if st.session_state.lang == "id" else 1
    
    selected_lang_name = st.selectbox(
        t["lang_label"], 
        options=lang_options, 
        index=current_lang_idx
    )
    
    # Deteksi perubahan bahasa untuk memicu refresh teks UI
    new_lang_code = "id" if selected_lang_name == "Bahasa Indonesia" else "en"
    if new_lang_code != st.session_state.lang:
        st.session_state.lang = new_lang_code
        st.rerun()  # Refresh halaman agar teks langsung berganti bahasa
    
    st.divider()
    
    # Input Data Knowledge Base
    arxiv_input = st.text_input(t["arxiv_label"], value="1706.03762")
    
    if st.button(t["btn_init"], use_container_width=True):
        st.session_state.rag_engine = build_knowledge_base(arxiv_input)
        if st.session_state.rag_engine:
            st.session_state.agent_app = compile_agent_graph()
            st.success(t["success_init"])
            st.balloons()
            
    st.divider()
    
    # Utility Features
    st.subheader(t["sidebar_util"])
    if st.button(t["btn_clear"], use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("🤖 **Engine:** Qwen 2.5 (7B) via HuggingFace API<br>🧠 **Orchestrator:** LangGraph StateGraph", unsafe_allow_html=True)


# --- MAIN CHAT PANEL ---
st.title(t["title"])
st.markdown(t["welcome"])

# Cek kesiapan agen
if st.session_state.agent_app is None:
    st.info(t["info_init"])
else:
    st.success(t["status_ready"])

# Render Riwayat Obrolan Eksisting
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Menangani Input Baru dari Pengguna
if st.session_state.agent_app is not None:
    if prompt := st.chat_input(t["chat_placeholder"]):
        
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.chat_message("assistant"):
            final_answer = ""
            
            with st.status(t["status_thinking"], expanded=True) as status_box:
                for event in st.session_state.agent_app.stream({"messages": st.session_state.messages}):
                    if "agent" in event:
                        node_msg = event["agent"]["messages"][0]
                        st.session_state.messages.append(node_msg)
                        
                        if node_msg.tool_calls:
                            for tool_call in node_msg.tool_calls:
                                status_box.write(f"{t['status_tool']} `{tool_call['name']}`")
                                status_box.write(f"{t['status_arg']} `{tool_call['args']}`")
                        else:
                            final_answer = node_msg.content
                            
                    elif "tools" in event:
                        tool_msg = event["tools"]["messages"][0]
                        st.session_state.messages.append(tool_msg)
                        
                        preview = tool_msg.content[:200] + "..." if len(tool_msg.content) > 200 else tool_msg.content
                        status_box.write(t["status_output"])
                        status_box.code(preview, language="text")
                
                status_box.update(label=t["status_complete"], state="complete", expanded=False)
            
            if final_answer:
                st.markdown(final_answer)
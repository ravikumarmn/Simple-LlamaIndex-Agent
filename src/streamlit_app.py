import os
import json
import shutil
import streamlit as st
from langchain.agents import Tool
from document_manager import document_manager
from agent import (
    VectorDBTool,
    InappropriateContentDetector,
    get_agent
)

# Get API keys from secrets
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]
cohere_api_key = st.secrets["COHERE_API_KEY"]

# Set environment variables
os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["COHERE_API_KEY"] = cohere_api_key

@st.cache_resource
def get_document_manager():
    return document_manager

@st.cache_resource
def get_agent_instance():
    """Initialize and cache the agent instance"""
    try:
        with open("./config/ncert_search.json") as f:
            config = json.load(f)
        
        tools = [
            Tool(
                name="VectorDBTool",
                func=VectorDBTool().run,
                description="Use this tool to query the VectorDB for complex information."
            ),
            Tool(
                name="InappropriateContentDetector",
                func=InappropriateContentDetector().run,
                description="Detects inappropriate language in user queries and warns the user."
            )
        ]
        
        return get_agent(tools, config["agent_llm"])
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        return None

@st.cache_data
def process_uploaded_file(file_path):
    """Process and index uploaded file with caching"""
    doc_manager = get_document_manager()
    try:
        n_indexed = doc_manager.index_doc_from_files([file_path])
        return n_indexed, None
    except Exception as e:
        return 0, str(e)

# Add project information in sidebar
with st.sidebar:
    st.title("🤖 RAG Assistant")
    
    st.markdown("""
    <style>
    .sidebar-content {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .feature-header {
        color: #1f77b4;
        font-weight: bold;
        font-size: 1.2em;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="feature-header">📚 Key Features</p>', unsafe_allow_html=True)
    st.markdown("""
    - 📄 Upload documents (PDF, TXT)
    - 🔍 Document indexing
    - 💡 AI-powered Q&A
    - 🎯 Context-aware responses
    """)
    
    st.markdown('<p class="feature-header">🧠 Smart Agent System</p>', unsafe_allow_html=True)
    st.markdown("""
    - 🤖 Intelligent query processing
    - 🚫 Content safety monitoring
    """)

    st.markdown('<p class="feature-header">⚡ Tech Stack</p>', unsafe_allow_html=True)
    st.markdown("""
    - 🧠 **OpenAI**: GPT-4 integration
    - 🔍 **Pinecone**: Vector database
    - 📚 **LlamaIndex**: Document processing
    - 🎨 **Streamlit**: User interface
    """)

    with st.expander("🛠️ Agent Capabilities"):
        st.markdown("""
        Our smart agent system features:
        
        - **Query Analysis**: Smart question processing
        - **Document Retrieval**: Efficient information lookup
        - **Context Integration**: Seamless knowledge combination
        - **Safety Checks**: Content appropriateness verification
        """)
    
    st.divider()
    st.caption(" 2024 RAG Assistant")
    st.divider()

st.markdown(
    "<h1 style='text-align: center;'>Simple Agent System Conversation</h1>",
    unsafe_allow_html=True,
)

# Initialize the agent
agent = get_agent_instance()

# File uploader
uploaded_file = st.file_uploader("Choose a file to upload", type=["txt", "pdf"])

# Initialize session state
if "file_processed" not in st.session_state:
    st.session_state["file_processed"] = False

if uploaded_file is not None:
    folder_path = "./uploaded_files"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, uploaded_file.name)

    # Only write file if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, "wb") as output:
            shutil.copyfileobj(uploaded_file, output)

        # Process file only if newly uploaded
        n_indexed, error = process_uploaded_file(file_path)
        if error:
            st.error(f"An error occurred while indexing the documents: {error}")
        else:
            st.success(
                f"File uploaded and indexed successfully! {n_indexed} chunks created."
            )
            st.session_state["file_processed"] = True
    else:
        st.info("File already exists and has been processed.")
        st.session_state["file_processed"] = True

# Chat interface
if st.session_state["file_processed"] and agent:
    st.markdown("---")
    
    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = []

    for msg in st.session_state["agent_messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask something about your document..."):
        st.session_state["agent_messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Thinking..."):
            try:
                response = agent.invoke(prompt)
                if response and "output" in response:
                    response_text = response["output"]
                    st.session_state["agent_messages"].append(
                        {"role": "assistant", "content": response_text}
                    )
                    st.chat_message("assistant").write(response_text)
                else:
                    st.warning("No response available from the system.")
            except Exception as e:
                st.error(f"Failed to connect to agent: {str(e)}")
else:
    if not agent:
        st.error("Agent initialization failed. Please check your configuration and API keys.")
    else:
        st.info("Please upload a document to start asking questions!")

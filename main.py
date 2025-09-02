import streamlit as st
import time
from app.ui import pdf_uploader
from app.pdf_utils import extract_text_from_pdf
from app.vectorstore_utils import create_faiss_index, retrive_relevant_docs
from app.chat_utils import get_chat_model, ask_chat_model
from app.config import EURI_API_KEY
from langchain.text_splitter import RecursiveCharacterTextSplitter


st.set_page_config(
    page_title="MediChat Pro - Medical Document Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat styling with avatars and buttons
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
        color: black;
    }
    .chat-message .avatar {
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 50%;
        margin-right: 0.75rem;
    }
    .chat-message .message {
        flex: 1;
    }
    .chat-message .timestamp {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 0.5rem;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: background-color 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #ff3333;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .doc-preview {
        max-height: 120px;
        overflow-y: auto;
        background: #fff;
        border: 1px solid #ddd;
        padding: 0.5rem;
        margin-bottom: 0.75rem;
        border-radius: 0.3rem;
        font-family: monospace;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_model" not in st.session_state:
    st.session_state.chat_model = None
if "processing_step" not in st.session_state:
    st.session_state.processing_step = ""

# Avatar URLs for chat
USER_AVATAR = "https://i.pravatar.cc/40?img=5"
ASSISTANT_AVATAR = "https://i.pravatar.cc/40?img=12"


# Title with animated emoji (can replace with gif if desired)
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #ff4b4b; font-size: 3rem; margin-bottom: 0.5rem;">🏥 MediChat Pro</h1>
    <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Your Intelligent Medical Document Assistant</p>
</div>
""", unsafe_allow_html=True)


# Sidebar - Document Upload
with st.sidebar:
    st.markdown("### 📁 Upload Medical Documents")
    st.markdown("Upload your medical PDFs to start chatting with MediChat Pro!")
    
    uploaded_files = pdf_uploader()
    
    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} document(s) uploaded")
        
        # Show preview snippets of uploaded documents
        with st.expander("Preview Uploaded Documents"):
            for idx, file in enumerate(uploaded_files):
                text = extract_text_from_pdf(file)
                snippet = text[:1000] + ("..." if len(text) > 1000 else "")
                st.markdown(f"**Document {idx+1}:** {file.name}")
                st.markdown(f"<div class='doc-preview'>{snippet}</div>", unsafe_allow_html=True)
        
        # Process documents button
        if st.button("🚀 Process Documents", type="primary"):
            st.session_state.processing_step = "Uploading PDFs"
            with st.spinner("Processing your medical documents step 1/5... Uploading PDFs"):
                time.sleep(0.5)
            
            # Extract text from all PDFs
            all_texts = []
            st.session_state.processing_step = "Extracting Text"
            with st.spinner("Step 2/5: Extracting text from PDFs..."):
                for file in uploaded_files:
                    text = extract_text_from_pdf(file)
                    all_texts.append(text)
                    time.sleep(0.2)  # simulate small delay
            
            # Split texts into chunks
            st.session_state.processing_step = "Splitting Text"
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = []
            with st.spinner("Step 3/5: Splitting documents into chunks..."):
                for text in all_texts:
                    chunks.extend(text_splitter.split_text(text))
                    time.sleep(0.2)
            
            # Create FAISS index
            st.session_state.processing_step = "Creating Index"
            with st.spinner("Step 4/5: Creating FAISS vector index..."):
                vectorstore = create_faiss_index(chunks)
                st.session_state.vectorstore = vectorstore
                time.sleep(0.5)
            
            # Initialize chat model
            st.session_state.processing_step = "Loading Model"
            with st.spinner("Step 5/5: Initializing chat model..."):
                chat_model = get_chat_model(EURI_API_KEY)
                st.session_state.chat_model = chat_model
                time.sleep(0.5)
            
            st.success("✅ Documents processed successfully!")
            st.balloons()
            st.session_state.processing_step = ""
    else:
        st.info("Upload PDFs here to get started.")


# Main chat interface header
st.markdown("### 💬 Chat with Your Medical Documents")


def render_chat_message(role, content, timestamp):
    avatar_url = USER_AVATAR if role == "user" else ASSISTANT_AVATAR
    bg_color = "#2b313e" if role == "user" else "#f0f2f6"
    text_color = "white" if role == "user" else "black"
    st.markdown(f"""
    <div style='display:flex; align-items:center; margin-bottom:15px;'>
        <img src="{avatar_url}" style='border-radius:50%; width:40px; height:40px; margin-right:12px;' />
        <div>
            <div style='background-color: {bg_color}; color: {text_color}; 
                        border-radius:12px; padding:12px; max-width:660px; font-size: 1rem; line-height: 1.4;'>
              {content}
            </div>
            <small style='opacity:0.5; font-size:0.8rem;'>{timestamp}</small>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Render chat message history with avatars
for msg in st.session_state.messages:
    render_chat_message(msg["role"], msg["content"], msg["timestamp"])


# Chat input box
if prompt := st.chat_input("Ask about your medical documents..."):
    timestamp = time.strftime("%I:%M %p").lstrip("0")
    # Append user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    
    render_chat_message("user", prompt, timestamp)
    
    if st.session_state.vectorstore and st.session_state.chat_model:
        # Show typing indicator
        placeholder = st.empty()
        with placeholder.container():
            st.markdown(f"""
            <div style='font-style:italic; color:#999;'>Assistant is typing...</div>
            """, unsafe_allow_html=True)
        
        # Retrieve relevant docs & generate response
        relevant_docs = retrive_relevant_docs(st.session_state.vectorstore, prompt)
        context = "\n\n".join(doc.page_content for doc in relevant_docs)
        
        system_prompt = f"""You are MediChat Pro, an intelligent medical document assistant.
Based on the following medical documents, provide accurate and helpful answers.
If the information is not in the documents, clearly state that.
When giving an answer, try to take help of the language model and give a full diagnosis of the problem.

Medical Documents:
{context}

User Question: {prompt}

Answer:"""
        response = ask_chat_model(st.session_state.chat_model, system_prompt)
        placeholder.empty()
        
        render_chat_message("assistant", response, timestamp)
        
        # Append assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": timestamp
        })
    else:
        render_chat_message("assistant", "⚠️ Please upload and process documents first!", timestamp)


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
    <p>🤖 Powered by Euri AI & LangChain | 🏥 Medical Document Intelligence</p>
</div>
""", unsafe_allow_html=True)

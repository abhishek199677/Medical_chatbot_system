import streamlit as st
from app.ui import pdf_uploader
from app.pdf_utils import extract_text_from_pdf
from app.vectorstore_utils import create_faiss_index, retrive_relevant_docs
from app.chat_utils import get_chat_model, ask_chat_model
from app.config import EURI_API_KEY
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

# Initialize session state variables at the top
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "chat_model" not in st.session_state:
    st.session_state["chat_model"] = None

st.set_page_config(
    page_title="MediChat Pro - Medical Document Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS and UI setup code here (unchanged)...

# Sidebar for document upload
with st.sidebar:
    st.markdown("### 📁 Document Upload")
    st.markdown("Upload your medical documents to start chatting!")
    
    uploaded_files = pdf_uploader()

    # Optionally reset vectorstore and chat_model if new files uploaded
    if uploaded_files:
        # You can reset vectorstore and chat_model if you want to reprocess new files
        # st.session_state["vectorstore"] = None
        # st.session_state["chat_model"] = None

        st.success(f"📄 {len(uploaded_files)} document(s) uploaded")

        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing your medical documents..."):
                all_texts = []
                for file in uploaded_files:
                    text = extract_text_from_pdf(file)
                    all_texts.append(text)

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                chunks = []
                for text in all_texts:
                    chunks.extend(text_splitter.split_text(text))

                vectorstore = create_faiss_index(chunks)
                st.session_state["vectorstore"] = vectorstore

                chat_model = get_chat_model(EURI_API_KEY)
                st.session_state["chat_model"] = chat_model
                
                st.success("✅ Documents processed successfully!")
                st.balloons()

# Main chat interface and message display (unchanged)...
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(message["timestamp"])

# Chat input handler
if prompt := st.chat_input("Ask about your medical documents..."):
    timestamp = time.strftime("%H:%M")

    # Append user message
    st.session_state["messages"].append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })

    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(timestamp)

    if st.session_state["vectorstore"] and st.session_state["chat_model"]:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):
                relevant_docs = retrive_relevant_docs(st.session_state["vectorstore"], prompt)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                system_prompt = f"""You are MediChat Pro, an intelligent medical document assistant. 
Based on the following medical documents, provide accurate and helpful answers. 
If the information is not in the documents, clearly state that.
when your giving medical advice always remind the user to consult a healthcare professional for accurate diagnosis and treatment and also give me thema the full diagnosis of the patient and also suggest them to go to the nearest hospital.

Medical Documents:
{context}

User Question: {prompt}

Answer:"""

                response = ask_chat_model(st.session_state["chat_model"], system_prompt)

            st.markdown(response)
            st.caption(timestamp)

            # Append assistant response
            st.session_state["messages"].append({
                "role": "assistant",
                "content": response,
                "timestamp": timestamp
            })
    else:
        with st.chat_message("assistant"):
            st.error("⚠️ Please upload and process documents first!")
            st.caption(timestamp)

# Footer (unchanged)...
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🤖 Powered by Euri AI & LangChain | 🏥 Medical Document Intelligence</p>
</div>
""", unsafe_allow_html=True)

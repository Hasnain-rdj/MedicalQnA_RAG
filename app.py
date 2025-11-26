"""
Medical RAG QA System - Streamlit Web Interface

A web-based interface for the Medical RAG QA System that allows users to ask
medical questions and receive contextually accurate answers with source citations.
"""

import streamlit as st
import pickle
from pathlib import Path
import time
import pandas as pd

# LangChain imports
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page configuration
st.set_page_config(
    page_title="Medical RAG QA System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Global improvements */
    .main {
        background-color: #ffffff;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #34495e;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Answer box - clean white with green accent */
    .answer-box {
        background-color: #ffffff;
        color: #2c3e50;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #27ae60;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    /* Source box - light gray with blue accent */
    .source-box {
        background-color: #f8f9fa;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    /* Source box headers */
    .source-box strong {
        color: #2c3e50;
    }
    
    /* Improve text readability */
    .stMarkdown {
        color: #2c3e50;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 0.3rem;
        font-weight: 500;
    }
    
    .stButton button:hover {
        background-color: #2980b9;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
OUTPUT_DIR = Path('output')
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

@st.cache_resource
def load_rag_system(retriever_k=4, temperature=0.3):
    """
    Load and initialize the RAG system components.
    This function is cached to avoid reloading on every interaction.
    """
    try:
        # Check if vector store exists
        vector_store_path = OUTPUT_DIR / 'faiss_medical_index'
        if not vector_store_path.exists():
            st.error(f"Vector store not found at {vector_store_path}. Please run the notebook first.")
            return None
        
        # Initialize embeddings
        with st.spinner("Loading embedding model..."):
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        # Load FAISS vector store
        with st.spinner("Loading vector store..."):
            vectorstore = FAISS.load_local(
                str(vector_store_path),
                embeddings,
                allow_dangerous_deserialization=True
            )
        
        # Initialize LLM
        with st.spinner("Initializing language model..."):
            llm = ChatOllama(
                model=LLM_MODEL,
                temperature=temperature,
            )
        
        # Create prompt template
        prompt_template = """You are a medical assistant that provides accurate, evidence-based answers to medical questions.

Instructions:
- Use the provided context from medical records as your primary source when available
- If the context is relevant, base your answer on it
- If the context doesn't contain sufficient information, use your medical knowledge to provide a helpful answer
- Include the medical specialty when relevant
- Be clear, concise, and accurate
- Always prioritize patient safety in your responses

Context from medical records:
{context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retriever_k}
        )
        
        # Create RAG chain using LCEL (LangChain Expression Language)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        qa_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | PROMPT
            | llm
            | StrOutputParser()
        )
        
        return qa_chain, vectorstore, retriever
    
    except Exception as e:
        st.error(f"Error loading RAG system: {str(e)}")
        return None

def format_sources(source_documents):
    """Format source documents for display"""
    sources_html = ""
    for i, doc in enumerate(source_documents, 1):
        specialty = doc.metadata.get('medical_specialty', 'Unknown')
        keywords = doc.metadata.get('keywords', 'N/A')[:100]
        content_preview = doc.page_content[:250].replace('\n', ' ')
        
        sources_html += f"""
        <div class="source-box">
            <strong>Source {i}: {specialty}</strong><br>
            <small><strong>Keywords:</strong> {keywords}</small><br>
            <small><strong>Preview:</strong> {content_preview}...</small>
        </div>
        """
    return sources_html

def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">🏥 Medical RAG QA System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask medical questions and get evidence-based answers with source citations</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model settings
        st.subheader("Model Configuration")
        retriever_k = st.slider(
            "Number of sources to retrieve",
            min_value=1,
            max_value=10,
            value=4,
            help="How many relevant document chunks to retrieve for context"
        )
        
        temperature = st.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Lower values (0.1-0.3) for factual responses, higher (0.6-0.8) for creative"
        )
        
        st.divider()
        
        # System info
        st.subheader("📊 System Info")
        st.info(f"""
        **LLM Model:** {LLM_MODEL}  
        **Embeddings:** {EMBEDDING_MODEL}  
        **Vector Store:** FAISS  
        **Dataset:** Medical Transcriptions
        """)
        
        st.divider()
        
        # Example questions
        st.subheader("💡 Example Questions")
        example_questions = [
            "What are the symptoms of heart failure?",
            "How is a colonoscopy performed?",
            "What medications treat hypertension?",
            "What is the recovery time for hip replacement?",
            "What are the warning signs of a stroke?"
        ]
        
        for q in example_questions:
            if st.button(q, key=f"example_{q[:20]}", use_container_width=True):
                st.session_state.question = q
        
        st.divider()
        
        # About section
        with st.expander("ℹ️ About"):
            st.markdown("""
            This Medical RAG QA System uses:
            - **Retrieval-Augmented Generation (RAG)** to provide accurate answers
            - **Local LLM** (Llama 3.2 via Ollama) for response generation
            - **FAISS** vector database for efficient similarity search
            - **Medical Transcriptions** dataset from Kaggle
            
            The system retrieves relevant medical context and generates
            evidence-based answers with proper source citations.
            """)
    
    # Main content area
    st.subheader("🔍 Ask Your Question")
    
    # Question input
    question = st.text_area(
        "Enter your medical question:",
        value=st.session_state.get('question', ''),
        height=100,
        placeholder="e.g., What are the symptoms of diabetes?",
        key="question_input"
    )
    
    # Action buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        submit_btn = st.button("🔍 Get Answer", type="primary", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_btn:
        st.session_state.question = ''
        st.rerun()
    
    # Process question
    if submit_btn and question.strip():
        # Load RAG system
        rag_result = load_rag_system(retriever_k, temperature)
        
        if rag_result is None:
            st.error("Failed to load RAG system. Please ensure:")
            st.markdown("""
            1. Ollama is running (`ollama list` to check)
            2. The notebook has been run to generate the vector store
            3. All required files are in the `output/` directory
            """)
            return
        
        qa_chain, vectorstore, retriever = rag_result
        
        # Get answer
        with st.spinner("🤔 Thinking... Retrieving relevant context and generating answer..."):
            start_time = time.time()
            try:
                # Get relevant documents first
                source_documents = retriever.invoke(question)
                
                # Get answer from chain
                answer = qa_chain.invoke(question)
                elapsed_time = time.time() - start_time
                
                # Display answer
                st.markdown("---")
                st.subheader("💡 Answer")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                
                # Display metadata
                col_meta1, col_meta2 = st.columns(2)
                with col_meta1:
                    st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
                with col_meta2:
                    st.caption(f"📚 Sources used: {len(source_documents)}")
                
                # Display sources
                st.markdown("---")
                st.subheader("📚 Source Documents")
                st.markdown(
                    format_sources(source_documents),
                    unsafe_allow_html=True
                )
                
                # Option to view raw sources
                with st.expander("🔬 View Full Source Content"):
                    for i, doc in enumerate(source_documents, 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(doc.page_content)
                        st.json(doc.metadata)
                        st.divider()
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                st.info("Please ensure Ollama is running with the llama3.2 model loaded.")
    
    elif submit_btn:
        st.warning("⚠️ Please enter a question before clicking 'Get Answer'")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>
            Medical RAG QA System | Built with LangChain, Ollama & Streamlit<br>
            ⚠️ For informational purposes only. Always consult healthcare professionals for medical advice.
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize session state
    if 'question' not in st.session_state:
        st.session_state.question = ''
    
    main()

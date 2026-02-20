"""
RAG Question Answering System - Web Interface
==============================================
Interactive web application for querying AI research papers using RAG.

Author: Azeem Khalipha
GitHub: https://github.com/azeemkhalipha
LinkedIn: https://www.linkedin.com/in/azeemkhalipha/
"""

import streamlit as st
import numpy as np
from typing import List, Dict
import time
import os
from google import genai
from google.genai import types
from PyPDF2 import PdfReader
import re

# Page configuration
st.set_page_config(
    page_title="RAG QA System - AI Research Papers",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chunk-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .answer-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# RAG SYSTEM COMPONENTS
# ============================================================================

class DocumentLoader:
    """Load and extract text from PDF documents"""
    
    @staticmethod
    def load_pdf(uploaded_file) -> str:
        """Extract text from uploaded PDF"""
        try:
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return ""


class TextChunker:
    """Split documents into overlapping chunks"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, doc_name: str) -> List[Dict]:
        """Split text into overlapping chunks"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append({
                        'id': f"{doc_name}_chunk_{chunk_id}",
                        'document': doc_name,
                        'text': current_chunk.strip(),
                        'chunk_num': chunk_id
                    })
                    chunk_id += 1
                    
                    words = current_chunk.split()
                    overlap_text = " ".join(words[-int(self.overlap/5):])
                    current_chunk = overlap_text + " " + sentence + " "
                else:
                    current_chunk = sentence + " "
        
        if current_chunk.strip():
            chunks.append({
                'id': f"{doc_name}_chunk_{chunk_id}",
                'document': doc_name,
                'text': current_chunk.strip(),
                'chunk_num': chunk_id
            })
        
        return chunks


class EmbeddingGenerator:
    """Generate embeddings with rate limiting"""
    
    def __init__(self, client, model_name: str = "models/gemini-embedding-001"):
        self.client = client
        self.model_name = model_name
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text
        )
        return np.array(result.embeddings[0].values)
    
    def batch_generate(self, texts: List[str], progress_bar=None) -> List[np.ndarray]:
        """Generate embeddings with progress tracking"""
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
            
            if progress_bar:
                progress_bar.progress((i + 1) / total)
            
            # Rate limiting
            if i < total - 1:
                time.sleep(0.65)
        
        return embeddings


class VectorStore:
    """In-memory vector store with cosine similarity search"""
    
    def __init__(self):
        self.chunks = []
        self.embeddings = []
    
    def add_chunks(self, chunks: List[Dict], embeddings: List[np.ndarray]):
        """Add chunks and embeddings"""
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """Search for most similar chunks"""
        similarities = []
        for chunk, embedding in zip(self.chunks, self.embeddings):
            similarity = self.cosine_similarity(query_embedding, embedding)
            similarities.append((chunk, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class AnswerGenerator:
    """Generate answers using LLM"""
    
    def __init__(self, client, model: str = "models/gemini-2.5-flash"):
        self.client = client
        self.model = model
    
    def create_prompt(self, query: str, chunks: List[Dict]) -> str:
        """Create grounded prompt"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['document']}, Chunk {chunk['chunk_num']}]\n"
                f"{chunk['text']}\n"
                f"[Relevance: {chunk['relevance_score']:.3f}]\n"
            )
        
        context = "\n".join(context_parts)
        
        return f"""You are a helpful research assistant answering questions based on AI research papers.

TASK: Answer the following question using ONLY the information provided in the context below.

INSTRUCTIONS:
1. Base your answer ONLY on the provided context
2. Cite sources using the format [Source X]
3. If insufficient information, say "I don't have enough information in the provided context to answer this question."
4. Be concise but comprehensive
5. Do not use training data - only provided context

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    
    def generate_answer(self, query: str, chunks: List[Dict]) -> Dict:
        """Generate answer with citations"""
        prompt = self.create_prompt(query, chunks)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
            )
        )
        
        return {
            'question': query,
            'answer': response.text,
            'sources': [f"{c['document']} (Chunk {c['chunk_num']})" for c in chunks],
            'retrieved_chunks': chunks
        }


# ============================================================================
# STREAMLIT APP
# ============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'embedding_gen' not in st.session_state:
        st.session_state.embedding_gen = None
    if 'answer_gen' not in st.session_state:
        st.session_state.answer_gen = None
    if 'all_chunks' not in st.session_state:
        st.session_state.all_chunks = []
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []


def main():
    """Main Streamlit application"""
    
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG Question Answering System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about AI research papers with intelligent retrieval and grounded answers</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google AI API Key",
            type="password",
            help="Get your API key from https://aistudio.google.com/app/apikey"
        )
        
        if api_key:
            st.success("✅ API Key configured")
            client = genai.Client(api_key=api_key)
        else:
            st.warning("⚠️ Please enter your API key to continue")
            st.stop()
        
        st.divider()
        
        # Document upload
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF research papers",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload 3-4 AI research papers"
        )
        
        # Process documents button
        if uploaded_files and st.button("🚀 Process Documents", type="primary"):
            process_documents(uploaded_files, client)
        
        st.divider()
        
        # System stats
        if st.session_state.documents_loaded:
            st.header("📊 System Stats")
            st.metric("Documents Loaded", len(uploaded_files))
            st.metric("Total Chunks", len(st.session_state.all_chunks))
            st.metric("Embeddings Generated", len(st.session_state.all_chunks))
            st.metric("Questions Asked", len(st.session_state.qa_history))
        
        st.divider()
        
        # About section
        st.header("ℹ️ About")
        st.markdown("""
        This system uses **Retrieval-Augmented Generation (RAG)** to answer questions about research papers.
        
        **How it works:**
        1. 📥 Upload research papers
        2. ✂️ Split into chunks
        3. 🧠 Generate embeddings
        4. 🔍 Semantic search
        5. 💬 LLM generates answer
        
        **Models:**
        - Embeddings: `gemini-embedding-001`
        - Generation: `gemini-2.5-flash`
        """)
        
        # Footer
        st.divider()
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        Built with Streamlit & Google Gemini<br>
        <a href='[Your GitHub]'>GitHub</a> | 
        <a href='[Your LinkedIn]'>LinkedIn</a>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if not st.session_state.documents_loaded:
        # Welcome screen
        st.info("👈 Upload research papers in the sidebar to get started!")
        
        # Example questions
        st.subheader("📝 Example Questions You Can Ask:")
        example_questions = [
            "What are the key components of the Transformer architecture?",
            "How does multi-head attention work?",
            "What is the purpose of positional encoding?",
            "Explain the concept of few-shot learning in GPT-3",
            "What are the main components of a RAG model?"
        ]
        
        for i, q in enumerate(example_questions, 1):
            st.markdown(f"{i}. *{q}*")
        
        # Features showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h3>🎯 Accurate</h3>
            <p>Grounded answers with source citations</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h3>⚡ Fast</h3>
            <p>Semantic search in milliseconds</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
            <h3>🔗 Transparent</h3>
            <p>Full source attribution</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Question answering interface
        st.subheader("💬 Ask Questions")
        
        # Question input
        user_question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the Transformer architecture?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("🔍 Ask", type="primary")
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.qa_history = []
                st.rerun()
        
        if ask_button and user_question:
            answer_question(user_question, client)
        
        # Display QA history
        if st.session_state.qa_history:
            st.divider()
            st.subheader("📜 Question History")
            
            for i, qa in enumerate(reversed(st.session_state.qa_history), 1):
                with st.expander(f"Q{len(st.session_state.qa_history) - i + 1}: {qa['question']}", expanded=(i==1)):
                    # Answer
                    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                    st.markdown(f"**Answer:**\n\n{qa['answer']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Sources
                    st.markdown("**Sources:**")
                    for source in qa['sources']:
                        st.markdown(f'<div class="source-box">📄 {source}</div>', unsafe_allow_html=True)
                    
                    # Retrieved chunks
                    with st.expander("🔍 View Retrieved Chunks"):
                        for j, chunk in enumerate(qa['retrieved_chunks'], 1):
                            st.markdown(f"""
                            <div class="chunk-box">
                            <strong>[{j}] {chunk['document']} - Chunk {chunk['chunk_num']}</strong><br>
                            <em>Relevance: {chunk['relevance_score']:.3f}</em><br><br>
                            {chunk['text'][:300]}...
                            </div>
                            """, unsafe_allow_html=True)


def process_documents(uploaded_files, client):
    """Process uploaded documents"""
    
    with st.spinner("Processing documents... This may take a few minutes."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Load documents
        status_text.text("📥 Loading documents...")
        documents = {}
        for i, file in enumerate(uploaded_files):
            doc_name = file.name.replace('.pdf', '')
            text = DocumentLoader.load_pdf(file)
            documents[doc_name] = text
            progress_bar.progress((i + 1) / (len(uploaded_files) * 3))
        
        # Step 2: Chunk documents
        status_text.text("✂️ Chunking documents...")
        chunker = TextChunker()
        all_chunks = []
        for i, (doc_name, text) in enumerate(documents.items()):
            chunks = chunker.chunk_text(text, doc_name)
            all_chunks.extend(chunks)
            progress_bar.progress((len(uploaded_files) + i + 1) / (len(uploaded_files) * 3))
        
        # Step 3: Generate embeddings
        status_text.text(f"🧠 Generating embeddings for {len(all_chunks)} chunks... (This takes ~{len(all_chunks) * 0.65 / 60:.1f} minutes)")
        embedding_gen = EmbeddingGenerator(client)
        
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        
        # Create sub-progress for embeddings
        embedding_progress = st.progress(0)
        embeddings = embedding_gen.batch_generate(chunk_texts, embedding_progress)
        
        progress_bar.progress(1.0)
        
        # Step 4: Build vector store
        status_text.text("🔗 Building vector index...")
        vector_store = VectorStore()
        vector_store.add_chunks(all_chunks, embeddings)
        
        # Initialize other components
        answer_gen = AnswerGenerator(client)
        
        # Store in session state
        st.session_state.documents_loaded = True
        st.session_state.vector_store = vector_store
        st.session_state.embedding_gen = embedding_gen
        st.session_state.answer_gen = answer_gen
        st.session_state.all_chunks = all_chunks
        
        status_text.text("")
        progress_bar.empty()
        embedding_progress.empty()
        
        st.success(f"✅ Successfully processed {len(documents)} documents with {len(all_chunks)} chunks!")
        time.sleep(1)
        st.rerun()


def answer_question(question: str, client):
    """Answer a user question"""
    
    with st.spinner("🔍 Searching and generating answer..."):
        # Retrieve
        query_embedding = st.session_state.embedding_gen.generate_embedding(question)
        results = st.session_state.vector_store.search(query_embedding, top_k=5)
        
        retrieved_chunks = [
            {
                'chunk_id': chunk['id'],
                'document': chunk['document'],
                'chunk_num': chunk['chunk_num'],
                'text': chunk['text'],
                'relevance_score': float(score)
            }
            for chunk, score in results
        ]
        
        # Generate answer
        result = st.session_state.answer_gen.generate_answer(question, retrieved_chunks)
        
        # Add to history
        st.session_state.qa_history.append(result)
        
    st.success("✅ Answer generated!")
    st.rerun()


if __name__ == "__main__":
    main()

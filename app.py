# =============================================================================
# Medical Chatbot — RAG Pipeline (Pinecone + HuggingFace + Groq)
# =============================================================================

# ── 1. Environment Setup ─────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Single load_dotenv call at the top — adjust path if .env is one level up
load_dotenv(dotenv_path=".env", override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")  # consistent key name

if not PINECONE_API_KEY:
    raise EnvironmentError("PINECONE_API_KEY not found in environment.")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found in environment.")


# ── 2. PDF Loading ────────────────────────────────────────────────────────────
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

def load_pdf_files(data_dir: str):
    """Load all PDFs from the given directory."""
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


# ── 3. Document Filtering ─────────────────────────────────────────────────────
from typing import List
from langchain_core.documents import Document

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Strip all metadata except 'source' to keep documents lean."""
    return [
        Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source", "unknown")}
        )
        for doc in docs
    ]


# ── 4. Text Splitting ─────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_documents(docs)


# ── 5. Embeddings ─────────────────────────────────────────────────────────────
# Using HuggingFace (dimension=384). If you switch to OpenAI (dimension=1536),
# you must also recreate the Pinecone index with dimension=1536.

from langchain_huggingface import HuggingFaceEmbeddings  # not deprecated

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Uncomment below and comment out above to use OpenAI embeddings instead:
# from langchain_openai import OpenAIEmbeddings
# def get_embeddings():
#     return OpenAIEmbeddings(api_key=OPENAI_API_KEY)  # dimension=1536


# ── 6. Pinecone Vector Store ──────────────────────────────────────────────────
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

INDEX_NAME  = "medical-chatbot"
EMBED_DIM   = 384   # must match your embedding model's output dimension

def get_or_create_pinecone_index(pc: Pinecone):
    """Create the Pinecone index if it doesn't exist, then return it."""
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    # Always assigned — not scoped inside the if-block
    return pc.Index(INDEX_NAME)


def build_vector_store(text_chunks: List[Document], embeddings) -> PineconeVectorStore:
    """Embed chunks and upsert into Pinecone."""
    return PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )


def load_vector_store(embeddings) -> PineconeVectorStore:
    """Load an existing Pinecone index (no re-embedding)."""
    return PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )


# ── 7. Groq LLM ───────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq

def get_llm():
    # Valid Groq-hosted models: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )


# ── 8. RAG Chain ──────────────────────────────────────────────────────────────
# ── 8. RAG Chain ──────────────────────────────────────────────────────────────
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

PROMPT_TEMPLATE = """
You are a knowledgeable and empathetic medical assistant. Use the context below
to answer the question. If the answer is not in the context, say you don't know
rather than guessing.

Context:
{context}

Question: {question}

Answer:
"""

def build_rag_chain(vector_store, llm):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ── 9. Main Pipeline ──────────────────────────────────────────────────────────
def get_vector_store(pc, embeddings):
    index = get_or_create_pinecone_index(pc)
    stats = index.describe_index_stats()

    if stats['total_vector_count'] > 0:
        print(f"  Found {stats['total_vector_count']} existing vectors. Skipping upload.")
        return PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings
        )
    else:
        print("  Index empty. Loading and uploading documents...")
        raw_docs = load_pdf_files("data")
        minimal  = filter_to_minimal_docs(raw_docs)
        chunks   = split_documents(minimal)
        print(f"  {len(raw_docs)} documents → {len(chunks)} chunks")
        return build_vector_store(chunks, embeddings)

# ── Initialize once at startup ─────────────────────────────────────────────
print("Initializing pipeline...")
embeddings   = get_embeddings()
pc           = Pinecone(api_key=PINECONE_API_KEY)
vector_store = get_vector_store(pc, embeddings)
llm          = get_llm()
chain        = build_rag_chain(vector_store, llm)
print("Ready!")

# ── Flask routes ───────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data  = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    result = chain.invoke(query)
    return jsonify({"answer": result, "sources": []})

if __name__ == '__main__':
    app.run(debug=False, port=5000)
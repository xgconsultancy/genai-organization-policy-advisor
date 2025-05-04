import os
import re
import streamlit as st
from difflib import SequenceMatcher
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import ollama

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
VECTORSTORE_DIR = "vectorstore"
CACHED_DOCS_DIR = "cached_docs"

POLICY_TOPICS = {
    "HR Guidelines": "👥 Human Resources & Workplace Policies",
    "Data Security": "🔒 Data-Security Protocols",
    "Health & Safety": "⚕️ Health & Safety Regulations"
}

DESCRIPTIONS = {
    "HR Guidelines": (
        "Upload your HR policy docs—employee handbooks, codes of conduct, appraisal processes—"
        "and ask about recruitment rules, leave entitlements, disciplinary procedures, etc."
    ),
    "Data Security": (
        "Upload data-security standards, GDPR policies, ISO27001 docs, and get instant guidance"
        " on encryption requirements, breach protocols, access controls, and more."
    ),
    "Health & Safety": (
        "Upload your H&S manuals, risk assessments, COSHH regs, and ask about fire safety,"
        " first aid procedures, PPE requirements, and all that."
    )
}

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def deduplicate_text(text, threshold=0.8):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    deduped_sentences = []
    for sentence in sentences:
        clean_sentence = sentence.strip()
        if not clean_sentence:
            continue
        duplicate_found = False
        for existing in deduped_sentences:
            if SequenceMatcher(None, clean_sentence, existing).ratio() > threshold:
                duplicate_found = True
                break
        if not duplicate_found:
            deduped_sentences.append(clean_sentence)
    return " ".join(deduped_sentences)

def load_and_chunk_docs(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    st.write(f"Created {len(chunks)} policy chunks from the document.")
    return chunks

def summarize_with_ollama(text, question):
    prompt = f"""
You are a helpful assistant. 
Summarize the answer to the following question strictly based ONLY on the provided policy document text. 
Avoid duplication and do not invent information.

Question: {question}

Document:
{text}

Answer:
"""
    response = ollama.chat(
        model='mistral',
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content'].strip()

# ---------------------------------------------------------------------
# Streamlit Layout
# ---------------------------------------------------------------------
st.set_page_config(page_title="Policy Advisor", page_icon="📄")
st.sidebar.title("Choose Policy Type")
topic = st.sidebar.selectbox("Select a Policy Domain", list(POLICY_TOPICS.keys()))

st.title(f"{topic} Advisor")
st.subheader(POLICY_TOPICS[topic])
st.write(DESCRIPTIONS[topic])

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = None
loaded_existing = False

# Load existing vectorstore
if os.path.exists(VECTORSTORE_DIR):
    try:
        db = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
        st.success(f"Loaded existing {topic} policy documents!")
        loaded_existing = True
    except Exception as e:
        st.error(f"Error loading policy documents: {e}")

# Upload new PDF
st.write("### Upload Your Policy Document")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], key="primary_pdf")

if uploaded_file is not None:
    os.makedirs(CACHED_DOCS_DIR, exist_ok=True)
    file_path = os.path.join(CACHED_DOCS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.write(f"Saved '{uploaded_file.name}' locally. Processing...")

    chunks = load_and_chunk_docs(file_path)
    if chunks:
        if db and loaded_existing:
            db.add_documents(chunks)
            db.save_local(VECTORSTORE_DIR)
            st.success("Added new policy document to knowledge base!")
        else:
            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(VECTORSTORE_DIR)
            st.success("Created a new policy knowledge base!")
    else:
        st.error("Failed to extract text from PDF.")

# Question Answering
if db is not None:
    st.write("---")
    st.subheader("Ask Your Policy Questions")

    query = st.text_input("Type your policy-related question:")
    if query:
        with st.spinner("Searching policies..."):
            retriever = db.as_retriever(search_kwargs={"k": 5})
            docs = retriever.get_relevant_documents(query)

            if not docs:
                st.warning("No relevant information found in documents.")
            else:
                merged_text = "\n".join([doc.page_content for doc in docs])
                merged_text = deduplicate_text(merged_text)
                answer = summarize_with_ollama(merged_text, query)

                st.markdown(f"**Answer:** {answer}")

                with st.expander("Sources"):
                    for idx, doc in enumerate(docs, start=1):
                        st.markdown(f"**Source {idx}:** {doc.metadata.get('source', 'Unknown')}")
else:
    st.info("Please upload a policy document to begin.")
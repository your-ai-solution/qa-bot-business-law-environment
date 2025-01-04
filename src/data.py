import os
import re
import fitz  # PyMuPDF
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Paths
RAW_DATA_DIR = "data/raw"
PREPROCESSED_DATA_DIR = "data/preprocessed/"

# Ensure preprocessed data directory exists
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

def clean_text(text):
    """
    Clean extracted text by removing unnecessary whitespace, line breaks, and special characters.
    """
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces and line breaks
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF file and cleans it.
    """
    doc = fitz.open(pdf_path)  # Open the PDF
    full_text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Load the page
        full_text += page.get_text("text")  # Extract text
    doc.close()
    return clean_text(full_text)

def process_pdfs_to_documents(raw_data_dir):
    """
    Process PDFs into LangChain Document objects with metadata.
    """
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for file_name in os.listdir(raw_data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(raw_data_dir, file_name)
            print(f"Processing: {pdf_path}")
            report_text = extract_text_from_pdf(pdf_path)

            if len(report_text) < 100:
                print(f"Skipping {file_name}: insufficient content.")
                continue

            chunks = text_splitter.create_documents(
                [report_text],
                metadatas=[{"source": file_name}]
            )
            documents.extend(chunks)

    return documents

def build_faiss_index(documents):
    """
    Build a FAISS index from LangChain Document objects and save it.
    """
    print("Building FAISS index...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(PREPROCESSED_DATA_DIR)
    print(f"FAISS index and documents saved to: {PREPROCESSED_DATA_DIR}")

def process_pdfs_to_embeddings():
    """
    Main processing function to extract text, clean, chunk, and index documents.
    """
    print("Extracting and processing PDFs...")
    documents = process_pdfs_to_documents(RAW_DATA_DIR)
    print(f"Total documents processed: {len(documents)}")

    if not documents:
        print("No documents found for processing.")
        return

    build_faiss_index(documents)
    print("Processing and indexing completed successfully.")

def main():
    """
    Main entry point for the script.
    """
    print("\n=== Business Law QA Bot: PDF Preprocessing ===\n")
    process_pdfs_to_embeddings()
    print("\nPreprocessing completed. Ready for retrieval!\n")

if __name__ == "__main__":
    main()
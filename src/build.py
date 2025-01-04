import os
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Paths
PROCESSED_DATA_DIR = "data/preprocessed/"
HUGGINGFACE_KEY_FILE = "configs/huggingface_api_key.txt"
API_URL = "https://api-inference.huggingface.co/models/gpt2"

# Function to read the Hugging Face API key from the text file
def read_huggingface_api_key(key_file):
    """
    Reads the Hugging Face API key from a text file.
    """
    try:
        with open(key_file, "r") as file:
            api_key = file.readline().strip()
            return api_key
    except FileNotFoundError:
        print(f"Error: API key file not found at {key_file}")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading the API key: {e}")
        exit(1)

# Load the Hugging Face API key
HUGGINGFACE_API_KEY = read_huggingface_api_key(HUGGINGFACE_KEY_FILE)
headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}"
}

# Function to load FAISS index
def load_faiss_index(processed_data_dir):
    """
    Load the FAISS index and embedding model.
    """
    print(f"Loading FAISS index from: {processed_data_dir}")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        processed_data_dir,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded successfully.")
    return vector_store

# Function to send a query to Hugging Face's API
def query_huggingface_api(prompt):
    """
    Query the Hugging Face API to generate a response.
    """
    print("Querying Hugging Face API...")
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        print("Response received successfully.")
        return response.json()[0]["generated_text"]
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# Function to generate response using Hugging Face API
def generate_response_with_online_model(query, retrieved_chunks):
    """
    Generate a response from the Hugging Face API using retrieved context.
    """
    context = "\n\n".join([doc.page_content for doc in retrieved_chunks if doc.page_content])[:1500]
    prompt = (
        f"You are a legal expert specializing in business laws and the legal environment. "
        f"Using the following context, answer the question concisely and accurately.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    return query_huggingface_api(prompt)

# Function to retrieve an answer to the given query
def query_business_law_report(query):
    """
    Retrieve documents and generate a response to the given query.
    """
    try:
        vector_store = load_faiss_index(PROCESSED_DATA_DIR)
        print("Retrieving relevant documents...")
        retriever = vector_store.as_retriever()
        retrieved_chunks = retriever.get_relevant_documents(query)

        if not retrieved_chunks:
            print("No relevant documents found.")
            return None

        return generate_response_with_online_model(query, retrieved_chunks)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Main function to run the script
def main():
    """
    Interactive Q&A loop.
    """
    print("\nWelcome to the Business Law QA Bot!")
    print("You can ask questions about business laws and the legal environment.")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("Enter your question: ")
        if user_query.lower() == "exit":
            print("Exiting. Goodbye!")
            break

        print("\nProcessing your question...\n")
        answer = query_business_law_report(user_query)
        if answer:
            print(f"Answer:\n{answer}\n")
        else:
            print("Sorry, no answer could be generated.\n")

if __name__ == "__main__":
    main()
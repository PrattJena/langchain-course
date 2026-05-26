import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


if __name__ == "__main__":
    print("Ingesting....")
    loader = TextLoader("/Users/pratyushjena/Documents/langchain-course/mediumblog1.txt")
    documents = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks")

    # embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", output_dimensionality=1536)
    embeddings = OllamaEmbeddings(model="qwen3-embedding:latest", dimensions=1536)

    # Test embedding directly BEFORE sending to Pinecone
    test_texts = [doc.page_content for doc in texts]
    print(f"Number of text strings: {len(test_texts)}")

    print("Ingesting embeddings...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME")) 
    print("Ingestion complete!")   
    
    

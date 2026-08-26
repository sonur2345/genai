from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os


os.environ["GOOGLE_API_KEY"] = "AQ.Ab8RN6Kp_Pq389NBxe-HImHZ_N3fjL7J8R5CgAaHRTQPuCPo-g"

pdf_path = Path(__file__).parent / "data_structure.pdf"

#load this file python program

load_dotenv()
loader = PyPDFLoader( file_path = pdf_path)
docs = loader.load()

# Split the docs into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400,
)

chunks = text_splitter.split_documents(documents=docs)

#vector embedding chunks

# embedding_model = OpenAIEmbeddings(
#     model = "text-embedding-3-large"
# )

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

print("indexing of document done..")
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
import os

pdf_folder_path = "./data/pdfs/"

# Load PDF files
loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path) if fn.endswith('.pdf')]

all_documents = []

persist_dir = './vectorstore/'

embd = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

for loader in loaders:
    raw_document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
    docs = text_splitter.split_documents(raw_document)
    all_documents.extend(docs)  # Flatten the list of documents

vectorstore = Chroma.from_documents(
    documents=all_documents,
    embedding= embd,
    persist_directory=persist_dir
)
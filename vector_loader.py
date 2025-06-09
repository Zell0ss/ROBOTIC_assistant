from langchain.text_splitter import MarkdownTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import dotenv
dotenv.load_dotenv()
FAQ_CHROMA_PATH="chroma_data"
# MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
# MODEL_NAME="./models/all-MiniLM-L6-v2"
MODEL_NAME="./models/all-mpnet-base-v2"

#load myiam
with open("documents/myiam.txt", "r") as f:
    text = f.read()

# Chunk the text
splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=50)
chunks = splitter.split_text(text)

# Merge headers with content
documents = []
for chunk in chunks: 
    header = chunk.split('\n')[0]  # Assuming header is followed by /n
    documents.append(Document(page_content=chunk, metadata={"header": header, "category": "MyIAM"}))

# load PO
with open("documents/product_owner.txt", "r") as f:
    text = f.read()

# Chunk the text
splitter = MarkdownTextSplitter(chunk_size=1024, chunk_overlap=50)
chunks = splitter.split_text(text)

# Merge headers with content
for chunk in chunks:  
    header = chunk.split('\n')[0]
    documents.append(Document(page_content=chunk, metadata={"header": header, "category": "PO"}))


# Generate embeddings and store them on disk
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME, #local path
    model_kwargs={'device': 'cpu'}  # or 'cuda' if you have GPU 
)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=FAQ_CHROMA_PATH
)
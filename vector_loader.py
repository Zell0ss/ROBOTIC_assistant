from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import dotenv
dotenv.load_dotenv()
FAQ_CHROMA_PATH="chroma_data"

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
embeddings = OpenAIEmbeddings()
# vector_store = Chroma.from_texts(chunks, embeddings)
vector_store = Chroma.from_documents(documents, embeddings, persist_directory=FAQ_CHROMA_PATH)

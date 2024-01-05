from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain.embeddings import GPT4AllEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


pdf_path = "/Users/mghifary/Work/Code/AI/data/SSD_SISTER_BKD.pdf"
loader = PyMuPDFLoader(pdf_path)
data = loader.load()

chat_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
all_splits = text_splitter.split_documents(data)

# Define vectostore
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

# Define RetrievalQA chain
chain = RetrievalQA.from_chain_type(
    chat_llm, 
    retriever=vectorstore.as_retriever(),
    verbose=True
)

# Define prompt
query = "Apakah yang dimaksud dengan Penghitungan Angka Kredit?"

print(f"Query: {query}")
# docs = vectorstore.similarity_search(query)
# print(f"Docs (similarity search results): {docs}")

# Run the chain
response = chain({"query": query})






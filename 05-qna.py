from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

data = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

embeddings = OpenAIEmbeddings()

db = DocArrayInMemorySearch.from_documents(
    data, 
    embeddings
)

query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)

retriever = db.as_retriever()

llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")
# qdocs = "".join([docs[i].page_content for i in range(len(docs))])

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."

response = qa_stuff.run(query)
print(f"Response: {response}")

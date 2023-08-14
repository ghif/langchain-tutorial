from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import langchain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain


file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

data = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

"""
Define the Chain
"""
llm = ChatOpenAI(
    temperature=0.0,
    model_name="gpt-3.5-turbo"
)

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

# Set hardcoded exampels
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

"""
 Generate additional examples by LLM
"""
example_gen_chain = QAGenerateChain.from_llm(llm)

# Generate 5 additional examples
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)

# Exclude "qa_pairs" key
new_examples_v2 = [s["qa_pairs"] for s in new_examples]

examples += new_examples_v2 # combine the examples


"""
Manual evaluation
"""
langchain.debug = True # turn on debug mode
qa.run(examples[0]["query"])
langchain.debug = False # turn off debug mode

"""
LLLM assisted evaluation
"""
predictions = qa.apply(examples)
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(examples)

for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()
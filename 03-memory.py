from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory, 
    ConversationBufferWindowMemory, 
    ConversationTokenBufferMemory,
    ConversationSummaryBufferMemory
)

llm = ChatOpenAI(temperature=0.0)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

print(conversation.predict(input="Hi, my name is Andrew"))
print(conversation.predict(input="What is 1+1?"))
print(conversation.predict(input="What is my name?"))
print(f"Buffer: {memory.buffer}")

print(f"Load memory variables: {memory.load_memory_variables({})}")

memory = ConversationBufferMemory()
memory.save_context(
    {"input": "Hi"},
    {"output": "What's up"}
)

print(f"Buffer: {memory.buffer}")
print(f"Load memory variables: {memory.load_memory_variables({})}")
memory.save_context({"input": "Not much, just hanging"}, 
                    {"output": "Cool"})

print(f"Load memory variables: {memory.load_memory_variables({})}")


memory = ConversationBufferWindowMemory(k=1)
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
print(f"Load memory variables: {memory.load_memory_variables({})}")
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=False
)
print(conversation.predict(input="Hi, my name is Andrew"))
print(conversation.predict(input="What is 1+1?"))
print(conversation.predict(input="What is my name?"))

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=30)
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})

print(f"Load memory variables: {memory.load_memory_variables({})}")

# Create a long string
schedule = """
There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo.
"""

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})

memory.load_memory_variables({})

conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

print(conversation.predict(input="What would be a good demo to show?"))
print(f"Load memory variables: {memory.load_memory_variables({})}")
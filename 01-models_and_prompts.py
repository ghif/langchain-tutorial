from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Define model
chat = ChatOpenAI(temperature=0.0)
# print(chat)

# Create a prompt template
template_string = """
Translate the text that is delimited by triple backtiks into a style that is {style}. \
text: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

print(prompt_template.messages[0].prompt)
print(prompt_template.messages[0].prompt.input_variables)

customer_style = """
American English in a calm and respectful tone
"""

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

customer_messages = prompt_template.format_messages(
    style=customer_style,
    text=customer_email
)

print(f"customer messages content: {customer_messages[0].content}")
customer_response = chat(customer_messages)
print(f"customer response content: {customer_response.content}")


service_reply = """
Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""

service_style_pirate = """
a polite tone \
that speaks in English Pirate\
"""

service_messages = prompt_template.format_messages(
    style=service_style_pirate,
    text=service_reply
)

print(f"service message content: {service_messages[0].content}")
service_response = chat(service_messages)
print(f"service response content: {service_response.content}")
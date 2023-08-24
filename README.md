# Ngoprek LangChain
Repositori ini berisi hasil oprekan sederhana dalam membuat aplikasi berbasis generative AI dengan menggunakan [LangChain](https://github.com/langchain-ai/langchain) melalui bahasa pemograman Python.

## Apa itu LangChain?
> LangChain merupakan sebuah framework untuk membangun aplikasi yang didukung oleh Large Language Models (LLMs).

Saat ini LangChain dapat dioperasikan melalui [Python](https://python.langchain.com/docs/get_started/introduction.html) dan [JavaScript/TypeScript](https://js.langchain.com/docs/get_started/introduction/).
LangChain menawarkan setidaknya 2 nilai tambah dalam memudahkan pembangunan aplikasi dengan AI:
1. **Integrasi/interoperabilitas** antara LLM dengan sumber data eksternal (arsip, aplikasi atau API lain).
2. **Agency**: memungkinkan LLM untuk berinteraksi dengan lingkungan (*environment*) yang ada.

## Mengapa LangChain?
LLMs merupakan suatu teknologi AI transformatif yang memungkinkan kita sebagai pengembang perangkat lunak untuk membangun aplikasi-aplikasi inovatif. 
Namun, penggunaan LLMs dalam lingkugan yang terisolasi terkadang membatasi kemampuan yang sebenarnya. 
LangChain menawarkan kemudahan bagi pengembang untuk berinteraksi dengan LLMs serta menghubungkannya dengan sumber-sumber pengetahuan lain sehingga memaksimalkan potensi desain dan rekayasa *prompt*, melalui aspek-aspek berikut:

1. [**Components**](https://docs.langchain.com/docs/category/components): abstraksi modular yang menggambarkan cara berinteraksi dengan LLMs.
2. **Off-the-shelf chains**: perangkaian terstruktur dari komponen-komponen untuk menyelesaikan suatu permasalahan.

Dengan kedua aspek tersebut kita sudah dapat mengerjakan berbagai studi kasus.

## Instalasi

**LangChain**
```
pip install langchain
```

**Chainlit**
```
pip install chainlit
```

## Models
Models dapat dipandang sebagai antarmuka dari LLMs. Ada beberapa tipe Models yang disediakan LangChain.

### Language Model
**Menerima 1 teks** dan mengeluarkan 1 teks.
```python
from langchain.llms import OpenAI

llm = OpenAI()

prompt = "Bulan ini merupakan bulan Agustus, bulan depan merupakan bulan Desember. Apakah pertanyaan tersebut benar? Jika salah, jelaskan letak kesalahannya."
response = llm(prompt) # inference
```


### Chat Model
**Menerima rangkaian teks** dan mengeluarkan 1 teks.
```python
from langchain.chat_models import ChatOpenAI

chat_llm = ChatOpenAI()

prompt = "Bulan ini merupakan bulan Agustus, bulan depan merupakan bulan Desember. Apakah pertanyaan tersebut benar? Jika salah, jelaskan letak kesalahannya."
response = chat_llm.predict(prompt) # inference
```


## Prompt Template
Sebuah objek yang membantu mendesain prompt berdasarkan kombinasi dari input pengguna, template statis, dan informasi non-statis lainnya.

```python
from langchain.prompts import PromptTemplate

next_month = "Desember"
template = """
Bulan ini merupakan bulan Agustus, bulan depan merupakan bulan {month}. 
Apakah pertanyaan tersebut benar? Jika salah, jelaskan letak kesalahannya.
"""

prompt = PromptTemplate.from_template(template)
final_prompt = prompt.format(
    month=next_month
)
```

## Output Parser
Sebuah objek untuk menstrukturkan output dari LLM, e.g., dalam bentuk JSON.

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

first_schema = ResponseSchema(
    name="first",
    description="This is the first schema"
)

second_schema = ResponseSchema(
    name="first",
    description="This is the second schema"
)

output_parser = StructuredOutputParser(
    [first_schema, second_schema]
)

format_instruction = output_parser.get_format_instructions() # string
```

## Chains
Chains merupakan salah satu kemampuan terpenting yang dimiliki oleh LangChain, yaitu menggabungkan beberapa proses inferensi LLM menjadi suatu rangkaian untuk memecahkan masalah tertentu.

### 1. Simple Sequential Chains
Mengkombinasikan inferensi beberapa LLM menjadi sebuah rantai sekuensial: Chain 1 --> Chain 2 --> ....

```python
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Model
llm = ChatOpenAI()

# Chain 1
prompt1 = ChatPromptTemplate.from_template(...)
chain1 = LLMChain(llm=llm, prompt=prompt1)

# Chain 2
prompt2 = ChatPromptTemplate.from_template(...)
chain2 = LLMChain(llm=llm, prompt=prompt2)

overall_chain = SimpleSequentialChain(
    chains=[chain1, chain2], 
    verbose=True
)

response = overall_chain.run("<some texts>")
```

### 2. Sequential Chains
Mengkombinasikan inferensi LLM secara sekuensial seperti Simple Sequential Chain namun memungkinkan untuk multi-input atau multi-output.

```python
from langchain.chains import LLMChain, SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Model
llm = ChatOpenAI()

prompt1 = ChatPromptTemplate.from_template(
    """
    blah
    {input1}
    """
)
chain1 = LLMChain(
    llm=chat_llm,
    prompt=prompt1,
    output_key="output1",
    verbose=True
)

# Chain2: input=output1, output=output2
prompt2 = ChatPromptTemplate.from_template(
    """
    blah
    {output1}
    """
)
chain2 = LLMChain(
    llm=chat_llm,
    prompt=prompt2,
    output_key="output2",
    verbose=True
)

# Chain3: input=output1, output=output3
prompt3 = ChatPromptTemplate.from_template(
    """
    blah {output1}
    {indonesian_review}
    """
)
chain3 = LLMChain(
    llm=chat_llm,
    prompt=prompt3,
    output_key="output3",
    verbose=True
)

overall_seq_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["input1"],
    output_variables=["output1", "output2", "output3"],
    verbose=True
)
```

### 3. Retrieval QA
Sebuah objek chain yang dirancang khusus untuk Tanya-Jawab berdasarkan sumber informasi eksternal sebagai bagian dari prompt.

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

# Create vector/embeddings store DB
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(pages, embeddings)

chat_llm = ChatOpenAI()
chain = RetrievalQA.from_llm(
    llm=chat_llm,
    retriever=db.as_retriever()
)
```


## Embeddings dan Vector Store
*Embeddings* merupakan model deep learning untuk mengkonversi teks menjadi vektor berdimensi yang tetap, dimana hubungan kedekatan antar vektor dirancang agar memiliki makna semantik.

*Vector store* merupakan database tempat penyimpanan vektor-vektor dari *embeddings* untuk memudahkan pengelolaan dan pengambilan (*retrieval*) informasi.

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Define embeddings
embeddings = OpenAIEmbeddings()

# Define vector store DB
db = FAISS.from_documents(<original_texts>, embeddings)
```

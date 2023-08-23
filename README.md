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
```
pip install langchain
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



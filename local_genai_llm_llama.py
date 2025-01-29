from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import GPT4All
from langchain_community.vectorstores import Chroma
import chromadb
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

caminhos_pdf = [    './Manuals/using-integrations-oracle-integration-3.pdf',
                    './Manuals/SOASUITE.pdf',
                    './Manuals/SOASUITEHL7.pdf'
                    ]

pages = []
ids = []
counter = 1
for caminho_pdf in caminhos_pdf:
    doc_pages = PyPDFLoader(caminho_pdf).load_and_split()
    pages.extend(doc_pages)
    ids.append(str(counter))
    counter = counter + 1

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# load vector store
vectorstore = Chroma.from_documents(
    pages,
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

client = chromadb.PersistentClient(path="./Test")
collection = client.get_or_create_collection(name="test")

collection.add(
    documents=caminhos_pdf,
    ids=ids
)

gpt4all_path = '/Users/cristianohoshikawa/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf'

template = """
Se a query em questão não for comparativa entre SOA SUITE e OIC, considerar apenas os documentos pertinentes ao assunto, ou seja, 
se a pergunta for sobre SOA SUITE, considerar apenas os documentos de SOA SUITE. Se a pergunta for sobre OIC, considerar apenas o 
documento sobre OIC. Se a pergunta for comparativa entre SOA SUITE e OIC, considerar todos os documentos. Informe no inicio qual a 
ferramenta está sendo tratada: {input} 
"""

llm = GPT4All(model=gpt4all_path)

prompt = PromptTemplate.from_template(template)
chain = (
        {"context": retriever,
         "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

while (True):
    query = input()
    if query == "quit":
        break
    print(chain.invoke(query))

# Save GPT4All model
gpt4all_path = './gpt4all_model.yaml'
llm.save(gpt4all_path)

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import Chroma

def chat():

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

    llm = ChatOCIGenAI(
        model_id="meta.llama-3.1-405b-instruct",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.compartment.oc1..aaaaaaaaexpiw4a7dio64mkfv2t273s2hgdl6mgfvvyv7tycalnjlvpvfl3q",
        auth_profile="DEFAULT",  # replace with your profile name,
        model_kwargs={"temperature": 0.7, "top_p": 0.75, "max_tokens": 1000},
    )

    embeddings = OCIGenAIEmbeddings(
        model_id="cohere.embed-multilingual-v3.0",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.compartment.oc1..aaaaaaaaexpiw4a7dio64mkfv2t273s2hgdl6mgfvvyv7tycalnjlvpvfl3q",
        auth_profile="DEFAULT",  # replace with your profile name
    )

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

    template = """ 
    Se a query em questão não for comparativa entre SOA SUITE e OIC, considerar apenas os documentos pertinentes ao assunto, ou seja, 
    se a pergunta for sobre SOA SUITE, considerar apenas os documentos de SOA SUITE. Se a pergunta for sobre OIC, considerar apenas o 
    documento sobre OIC. Se a pergunta for comparativa entre SOA SUITE e OIC, considerar todos os documentos. Informe no inicio qual a 
    ferramenta está sendo tratada
    : {input} 
    """
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


def chat2():
    client = chromadb.PersistentClient(path="./Test")
    collection = client.get_collection(name="test")
    counter = 0
    pages = []
    while counter < collection.count():
        try:
            document = collection.get(str(counter + 1))["documents"][0]
            doc_pages = PyPDFLoader(document).load_and_split()
            pages.extend(doc_pages)
            counter = counter + 1
        except:
            print("End of Collection")
            break

    llm = ChatOCIGenAI(
        model_id="meta.llama-3.1-405b-instruct",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.compartment.oc1..aaaaaaaaexpiw4a7dio64mkfv2t273s2hgdl6mgfvvyv7tycalnjlvpvfl3q",
        auth_profile="DEFAULT",  # replace with your profile name,
        model_kwargs={"temperature": 0.7, "top_p": 0.75, "max_tokens": 1000},
    )

    embeddings = OCIGenAIEmbeddings(
        model_id="cohere.embed-multilingual-v3.0",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.compartment.oc1..aaaaaaaaexpiw4a7dio64mkfv2t273s2hgdl6mgfvvyv7tycalnjlvpvfl3q",
        auth_profile="DEFAULT",  # replace with your profile name
    )

    vectorstore = Chroma.from_documents(
        pages,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever()

    template = """ 
    Se a query em questão não for comparativa entre SOA SUITE e OIC, considerar apenas os documentos pertinentes ao assunto, ou seja, se a pergunta for sobre SOA SUITE, considerar apenas os documentos de SOA SUITE. Se a pergunta for sobre OIC, considerar apenas o documento sobre OIC. Se a pergunta for comparativa entre SOA SUITE e OIC, considerar todos os documentos. Informe no inicio qual a ferramenta está sendo tratada
    SOA SUITE, OIC: {input} 
    """
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

chat()


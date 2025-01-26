# Testando o OCI Generative AI

## Introdução

Apresento aqui um snippet da OCI Generative AI para consultas sobre funcionalidades do Oracle SOA SUITE e do Oracle Integration.

Ambas ferramentas são utilizadas hoje para estratégias de integração híbridas, ou seja, em ambientes Cloud e on-prem.

Como as ferramentas possuem funcionalidades e processos em comum, este código ajuda a entender melhor como executar uma mesma abordagem de integração em cada. Além disso, é possível explorar características comuns e diferenças.

### Pre-Requisitos

- Python 3.10 ou superior
- OCI CLI

### Instalando os pacotes Python

Instale os pacotes Python executando:

    pip install -r requirements.txt

### Teste

O código irá permitir você a escrever suas perguntas. Assim que você teclar ENTER, receberá a resposta baseada nos materiais lidos (arquivos PDF).

Basta executar:

    python oci_genai_llm.py --device="mps" --gpu_name="M2Max GPU 32 Cores"

![img.png](images/img.png)

## Referência

- [Extending SaaS by AI/ML features - Part 8: OCI Generative AI Integration with LangChain Use Cases](https://www.ateam-oracle.com/post/oci-generative-ai-integration-with-langchain-usecases)
- [Bridging cloud and conversational AI: LangChain and OCI Data Science platform](https://blogs.oracle.com/ai-and-datascience/post/cloud-conversational-ai-langchain-oci-data-science)

## Acknowledgments

- **Author** - Cristiano Hoshikawa (Oracle LAD A-Team Solution Engineer)
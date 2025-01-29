# Codigo adaptado para OCI Generative AI Agent
# A partir deste material
# https://wellsr.com/python/working-with-python-langchain-agents/

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

@tool
def divide(first_int: int, second_int: int) -> float:
    """Divide the first integer by the second integer."""
    return first_int * second_int


@tool
def subtract(first_int: int, second_int: int) -> float:
    "Subtract the second integer from the first integer."
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> float:
    "Exponentiate the base to the exponent power."
    return base**exponent


tools = [divide, subtract, exponentiate]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful mathematician. Answer the following question.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

llm = ChatOCIGenAI(
     model_id="cohere.command-r-16k",
     service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
     compartment_id="ocid1.compartment.oc1..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
     auth_profile="DEFAULT",  # replace with your profile name,
     model_kwargs={"temperature": 0.7, "top_p": 0.75, "max_tokens": 2000}
)

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({
    "input": "Take 2 to the power of 5 and divide the result obtained by the result of subtracting 8 from 24, then square the whole result"
}
)

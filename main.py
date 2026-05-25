from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()

class Source(BaseModel):
    """Schema used for source tracking"""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent's response"""
    answer: str = Field(description="The agent's answer to the user's question")
    sources: list[Source] = Field(description="List of source used to answer the question")

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches for information on the web.

#     Args:
#         query: Clean keyword search terms. DO NOT use search operators like 'site:'.

#     Returns:
#         Search results
#     """
#     print(f"Searching for: {query}")
#     return tavily.search(query=query)


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search for only 1 job posting for an ai engineer using langchain in the bay area on linkedin and list their details?"
                )
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()

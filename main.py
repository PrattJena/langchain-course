from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that searches for information on the web.

    Args:
        query: Clean keyword search terms. DO NOT use search operators like 'site:'.

    Returns:
        Search results
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)


llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
tools = [search]
agent = create_agent(model=llm, tools=tools)


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

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def search(query: str) -> str:
    """
    Tool that searches for information on the web.
    
    Args:
        query: The search query
        
    Returns:
        Search results
    """
    print(f"Searching for: {query}")
    return f"Tokyo weather is sunny"

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
tools = [search]
agent = create_agent(model=llm, tools=tools)



def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
    print(result)


if __name__ == "__main__":
    main()

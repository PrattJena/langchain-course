from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL_NAME = "qwen3.5:2b"

@tool
def get_product_price(product:str) -> float:
    """ Look up the price of a product from the inventory """
    print(f" > Executing tool get_product_price for {product}")
    prices = {
        "laptop": 1000.50,
        "mouse": 25.99,
        "keyboard": 75.50
    }
    return prices.get(product, 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available discount tiers: silver, gold, platinum.
    """
    print(f" > Executing apply_discount(price={price}, discount_tier='{discount_tier}')")

    discounts = {
        "silver": 0.05,
        "gold": 0.10,
        "platinum": 0.15,
    }

    discount = discounts.get(discount_tier.lower(), 0.0)
    return round(price * (1 - discount), 2)

@traceable(name="Agent React Loop")
def run_agent(query: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"ollama:{MODEL_NAME}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {query}")
    print("=" * 50)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant.\n\n"
                "The product inventory contains exactly these product names: "
                "laptop, mouse, keyboard.\n\n"
                "STRICT RULES:\n"
                "1. If the user asks for laptop, mouse, or keyboard, that is specific enough. "
                "Do not ask for a more specific product name.\n"
                "2. NEVER guess a product price. Always call get_product_price first.\n"
                "3. Only call apply_discount after get_product_price returns a price.\n"
                "4. Pass the exact returned price into apply_discount.\n"
                "5. If the user gives a discount tier, use that tier. "
                "If not, ask which tier to use.\n"
                "6. Do not calculate discounts yourself. Always use apply_discount."
            )
        ),
        HumanMessage(content=query),
    ]

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n --- Iteration {i} ---")
        ai_response = llm_with_tools.invoke(messages)
        
        tool_calls = ai_response.tool_calls
        
        if not tool_calls:
            print(f"\nFinal Answer: {ai_response.content}")
            return ai_response.content
        
        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print(f" [Tool Selected] {tool_name} with args {tool_args}")

        tool_to_use = tools_dict[tool_name]
        if tool_to_use not in tools:
            print(f"\nTool {tool_name} not found")
            return None
        
        observation = tool_to_use.invoke(tool_args)
        print(f" [Tool Response] : {observation}")
        
        messages.append(ai_response)
        messages.append(ToolMessage(content=observation, tool_call_id=tool_call_id))
    
    print("\nERROR: Max iterations reached")
    return None

if __name__ == "__main__":
    print("Langchain Agent")
    print()
    run_agent("What is the price of a laptop after applying a gold discount?")



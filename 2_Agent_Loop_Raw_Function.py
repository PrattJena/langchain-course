import trace
from dotenv import load_dotenv
from langsmith import traceable
import ollama

load_dotenv()

MAX_ITERATIONS = 10
MODEL_NAME = "qwen3.5:2b"

@traceable(run_type="tool")
def get_product_price(product:str) -> float:
    """ Look up the price of a product from the inventory """
    print(f" > Executing tool get_product_price for {product}")
    prices = {
        "laptop": 1000.50,
        "mouse": 25.99,
        "keyboard": 75.50
    }
    return prices.get(product, 0.0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available discount tiers: silver, gold, platinum.
    """
    print(f" > Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    price = float(price)
    discounts = {
        "silver": 0.05,
        "gold": 0.10,
        "platinum": 0.15,
    }

    discount = discounts.get(discount_tier.lower(), 0.0)
    return round(price * (1 - discount), 2)


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'mouse', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: platinum, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "float", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'platinum', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


@traceable(name="Ollama chat model", run_type="llm")
def ollama_chat_model(messages):
    return ollama.chat(model=MODEL_NAME, messages=messages, tools=tools_for_llm)


@traceable(name="Agent React Loop")
def run_agent(query: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {query}")
    print("=" * 50)

    messages = [
        {
            "role": "system",
            "content": (
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
        },
        {"role": "user", "content": query},
    ]

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n --- Iteration {i} ---")
        response = ollama_chat_model(messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls
        
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f" [Tool Selected] {tool_name} with args {tool_args}")

        tool_to_use = tools_dict[tool_name]
        if tool_to_use not in tools:
            print(f"\nTool {tool_name} not found")
            return None
        
        observation = tool_to_use(**tool_args)
        print(f" [Tool Response] : {observation}")
        
        messages.append(ai_message)
        messages.append({"role": "tool", "content": str(observation)})
    
    print("\nERROR: Max iterations reached")
    return None

if __name__ == "__main__":
    print("Langchain Agent")
    print()
    run_agent("What is the price of a laptop after applying a gold discount?")



import re
import inspect
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


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g., @traceable adds *, config=None)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES:
1. If the user asks for laptop, mouse, or keyboard, that is specific enough.Do not ask for a more specific product name.
2. NEVER guess a product price. Always call get_product_price first.
3. Only call apply_discount after get_product_price returns a price.
4. Pass the exact returned price into apply_discount.
5. If the user gives a discount tier, use that tier. If not, ask which tier to use.
6. Do not calculate discounts yourself. Always use apply_discount.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

@traceable(name="Ollama chat model", run_type="llm")
def ollama_chat_model(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


@traceable(name="Agent React Loop")
def run_agent(question: str):

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    print(f"Question: {question}")
    print("=" * 50)


    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n --- Iteration {i} ---")
        full_prompt = prompt + "\n" + scratchpad
        response = ollama_chat_model(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content
        print(f"LLM Output: {output}")

        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Final Answer Match] {final_answer_match}")
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer
        
        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(f"  [Parsing] No Action or Action Input found in LLM output")
            return None

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f" [Tool Selected] {tool_name} with args {tool_input_raw}")

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))
        print(f" [Tool Response] : {observation}")
        
        scratchpad += f"{output}\nObservation: {observation}\nThought:"
    
    print("\nERROR: Max iterations reached")
    return None

if __name__ == "__main__":
    print("Langchain Agent")
    print()
    run_agent("What is the price of a laptop after applying a gold discount?")



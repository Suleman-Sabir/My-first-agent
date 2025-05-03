import os
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    set_default_openai_client,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

def my_first_agent():
    # Load API key from environment
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

    # Setup OpenAI-compatible client using Gemini API
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    # Configure the agent environment
    set_default_openai_client(external_client)
    set_tracing_disabled(True)
    
    # Define model
    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client,
    )

    # Create agent
    agent = Agent(
        name="HelloAgent",
        instructions="Just respond with 'Hello, world!'",
        model=model,
    )

    # Run the agent with a dummy prompt
    result = Runner.run_sync(agent, "Say hello")
    
    # Print the output
    print("\n")
    print(result.final_output)
    print("\n")

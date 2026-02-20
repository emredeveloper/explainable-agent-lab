from pathlib import Path
from explainable_agent.config import Settings
from explainable_agent.openai_client import OpenAICompatClient
from explainable_agent.agent import ExplainableAgent
from explainable_agent.tools import define_tool

# ==========================================
# 1. Define a Custom Tool
# ==========================================
# Use the @define_tool decorator to easily add new capabilities to your agent.
# The agent will automatically detect it and parse its description and usage hints.

@define_tool(
    name="get_weather_info",
    description="Fetches current weather information for a specific city.",
    usage_hint="Input is just the city name, e.g., Tokyo or London"
)
def get_weather_info(city: str, _: Path) -> str:
    # In a real-world scenario, you would make an API call (e.g. OpenWeatherMap)
    # Here, we just mock a database.
    weather_db = {
        "tokyo": "Sunny, 22°C",
        "london": "Rainy, 14°C",
        "new york": "Cloudy, 16°C",
        "istanbul": "Clear, 25°C"
    }
    
    city_lower = city.lower().strip()
    result = weather_db.get(city_lower)
    
    if result:
        return f"The weather in {city} is currently {result}."
    else:
        # Intentionally raising an error to demonstrate the agent's self-healing capabilities!
        return f"ERROR: Weather information not found for city '{city}'. Please try another city."

# ==========================================
# 2. Run the Agent
# ==========================================
def main():
    settings = Settings.from_env()
    client = OpenAICompatClient(base_url=settings.base_url, api_key=settings.api_key)
    
    # Enable verbose to track how the agent uses the custom tool
    agent = ExplainableAgent(settings=settings, client=client, verbose=True)
    
    # Let's give the agent a task that requires our new custom tool
    task = "get_weather_info: Istanbul"
    print(f"\n[Starting Task] {task}\n")
    
    trace = agent.run(task)
    print("\nFinal Answer:", trace.final_answer)

if __name__ == "__main__":
    main()

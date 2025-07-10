import yaml
import dotenv
import os
from dotenv import load_dotenv
from src.orchestration.graph_builder import graph

def main():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Accept user input for initial query or use config
    user_query = cfg.get("initial_query")
    thread_id = cfg.get("thread_id")
    model_name = cfg.get("model_name")

    # Start the workflow
    response = graph.invoke(
        {"messages": [{"role": "user", "content": user_query}]},
        {"configurable": {"thread_id": thread_id}}
    )
    # Print the final output (could be improved to stream or show intermediate results)
    print(response)

if __name__ == "__main__":
    main()
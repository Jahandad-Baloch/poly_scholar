from fastapi import FastAPI
from src.orchestration.graph_builder import graph

app = FastAPI()

@app.post("/invoke")
async def invoke_agent(user_input: dict):
    response = graph.invoke(user_input)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
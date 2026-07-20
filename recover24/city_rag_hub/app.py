from fastapi import FastAPI
from src.logic.nodes.retrieve import simple_search
import uvicorn


app = FastAPI()

@app.get("/")
def home():
    return {"message: 일본 관광지검색 서버 준비완료"}

@app.get("/ask")
def ask(query: str):
    results = simple_search(query)
    import json
    json_data = results.to_json(orient="records", force_ascii=False)
    data = json.loads(json_data)
    
    return {
        "query": query,
        "results": data,
        "snapshot_as_of": "2026-01-10"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port = 8000)

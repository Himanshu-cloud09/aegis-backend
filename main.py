from fastapi import FastAPI

app = FastAPI(title="AEGIS API")


@app.get("/")
def home():
    return {
        "status": "AEGIS API is live!"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }

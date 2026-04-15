import os
from fastapi import FastAPI, UploadFile, BackgroundTasks, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from rag_engine import RAGEngine
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

engine = RAGEngine()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Query(BaseModel):
    question: str

@app.post("/upload")
@limiter.limit("5/minute")
async def upload(request: Request, file: UploadFile, background_tasks: BackgroundTasks):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    
    background_tasks.add_task(engine.process_document, file_location, os.path.splitext(file.filename)[1])
    return {"status": "Processing started", "filename": file.filename}

@app.post("/ask")
@limiter.limit("10/minute")
async def ask(request: Request, query: Query):
    context = engine.retrieve(query.question)
    
    if not context:
        return {"answer": "No relevant context found. Please upload a document first."}

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"Answer based on context: {' '.join(context)}"},
            {"role": "user", "content": query.question}
        ],
        model="llama-3.3-70b-versatile", 
    )

    return {"answer": response.choices[0].message.content}
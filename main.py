from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from together import Together
import os
import logging
from collections import defaultdict

# Initialize FastAPI app
app = FastAPI()
logger = logging.getLogger("uvicorn")

# Thread-safe chat history storage
chat_sessions = defaultdict(list)

# Set up Together API client
API_KEY = os.getenv("TOGETHER_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Set TOGETHER_API_KEY as an environment variable.")

client = Together(api_key=API_KEY)

# Request model
class QueryRequest(BaseModel):
    user_id: str  # Add user_id to track sessions
    query: str

@app.get("/")  
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.post("/chat")
async def chat(request: QueryRequest):
    """Handles chat queries with session memory."""
    try:
        user_id = request.user_id  # Extract user_id

        # Initialize chat history if new user
        if not chat_sessions[user_id]:  
            chat_sessions[user_id].append(
                {"role": "system", "content": "You are a helpful AI that remembers previous conversations."}
            )

        chat_sessions[user_id].append({"role": "user", "content": request.query})

        response = client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf",
            messages=chat_sessions[user_id],
        )

        ai_response = response.choices[0].message.content
        chat_sessions[user_id].append({"role": "assistant", "content": ai_response})

        return {"response": ai_response}

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

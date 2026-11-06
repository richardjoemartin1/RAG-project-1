import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llm_handler import LlamaChat, run_rag  # ✅ Fixed import
from ingest_data import ingest_website_content as load_zco_docs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#This disables parallelism in the HuggingFace tokenizers library, which is often used under the hood by OpenAI, HuggingFace models, or LangChain. It is mainly used to suppress warnings like:
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using tokenizers before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Chat request model
class ChatRequest(BaseModel):
    query: str
    session_id: str

# RAG-specific chat input model
class ChatInput(BaseModel):
    message: str
    session_id: str

# Health check
@app.get("/")
async def root():
    return {"message": "✅ LLM Chat App with Redis + RAG is running"}

# Redis-based contextual chat
@app.post("/ask/")
async def chat(request: ChatRequest):
    try:
        print(f"[Chat] Query: {request.query} | Session: {request.session_id}")
        chat_instance = LlamaChat(request.session_id)
        result = chat_instance.chat(request.query)
        return {"response": result["message"]}
    except Exception as e:
        return {"error": f"❌ Chat error: {str(e)}"}

# Get Redis chat history
@app.get("/history/{session_id}")
async def get_history(session_id: str):
    try:
        chat_instance = LlamaChat(session_id)
        history = chat_instance.get_chat_history()
        return {"session_id": session_id, "history": history}
    except Exception as e:
        return {"error": f"❌ History error: {str(e)}"}

# Trigger ChromaDB ingestion
@app.post("/load-zco-pages")
async def load_pages():
    try:
        load_zco_docs()
        return {"message": "✅ Website content loaded into Chroma DB"}
    except Exception as e:
        return {"error": f"❌ Ingestion error: {str(e)}"}

# RAG + LLM endpoint (no memory)
# @app.post("/rag-chat")
# async def rag_chat(item: ChatInput):
#     try:
#         print(f"[RAG] Message: {item.message} | Session: {item.session_id}")
#         response = run_rag(item.message)
#         return {"response": response}
#     except Exception as e:
#         return {"error": f"❌ RAG error: {str(e)}"}


#This code  was used before using LangChain methods

# @app.post("/ask")
# async def chat(request: ChatRequest):   
#     session_key = f"chat:{request.session_id}"
#     print("Session Key:", session_key)
#     user_message = {"role": "user", "content": request.query}
#     history_str = await redis_client.get(session_key)
#     chat_history = json.loads(history_str) if history_str else []
#     # Append user message
#     chat_history.append(user_message)
#     # Send to Ollama
#     try:
#         response = requests.post(
#             "http://localhost:11434/api/chat",
#             json={
#                 "model": "llama3.1:8b",  # or "llama3:8b" based on how you named it
#                 "messages": chat_history,
#                 "stream": False
#             }
#         )
#         response_data = response.json()
#         assistant_reply = response_data["message"]["content"]

#         # Append assistant response
#         chat_history.append({"role": "assistant", "content": assistant_reply})

#         # Save updated history to Redis
#         await redis_client.set(session_key, json.dumps(chat_history))

#         return {"response": assistant_reply}
#     except Exception as e:
#         return {"error": str(e)}








#Since you’re aiming for chat with memory (multi-turn conversations), we’ll switch from POST /api/generate to POST /api/chat, which allows sending chat history (messages) to Ollama’s local server. 
   
#this code is when u dont need memory and you are running a local ollama server in another cmd (ollama run llama3.1:8b )
# @app.post("/ask/")
# async def ask_website(request: QueryRequest):
#     query = request.query

#     # Construct request to Ollama's local API
#     try:
#         ollama_response = requests.post(
#             "http://localhost:11434/api/generate",
#             json={
#                 "model": "llama3.1:8b",
#                 "prompt": query,
#                 "stream": False  # If you want to stream, handle that separately
#             }
#         )

#         if ollama_response.status_code != 200:
#             return {"answer": f"Ollama Error {ollama_response.status_code}: {ollama_response.text}"}

#         result = ollama_response.json()
#         answer = result.get("response", "[No response from model]")
#         return {"answer": answer}

#     except Exception as e:
#         return {"answer": f"Error contacting LLaMA model: {str(e)}"}


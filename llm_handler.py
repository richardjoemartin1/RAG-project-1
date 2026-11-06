# llm_handler.py
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_redis import RedisChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import AzureOpenAI
from typing import cast
load_dotenv()
print("DEBUG AZURE_OPENAI_ENDPOINT:", repr(os.getenv("AZURE_OPENAI_ENDPOINT")))
print("DEBUG AZURE_DEPLOYMENT_NAME:", repr(os.getenv("AZURE_DEPLOYMENT_NAME")))
print("DEBUG AZURE_API_KEY:", "SET" if os.getenv("AZURE_API_KEY") else "MISSING")


# Load from .env or fallback to hardcoded
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = cast(str, os.getenv("AZURE_DEPLOYMENT_NAME"))
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

if not AZURE_OPENAI_ENDPOINT:
    raise ValueError("AZURE_OPENAI_ENDPOINT is not set.")
if not AZURE_DEPLOYMENT_NAME:
    raise ValueError("AZURE_DEPLOYMENT_NAME is not set.")
if not AZURE_API_KEY:
    raise ValueError("AZURE_API_KEY is not set.")

# ✅ Initialize Azure OpenAI client
azure_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version="2025-01-01-preview"
)

# ✅ Constants
CHROMA_PERSIST_DIR = "./chroma_store"

# ✅ Initialize embedding model (used for similarity search)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ get_llm_response using Azure model
def get_llm_response(prompt: str) -> str:
    try:
        from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

        messages = [
            ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant."),
            ChatCompletionUserMessageParam(role="user", content=prompt)
        ]

        completion = azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=messages,
            max_completion_tokens=2000,
            temperature=1
        )

        return completion.choices[0].message.content or "" #

    except Exception as e:
        return f"❌ Azure OpenAI error: {str(e)}"

# ✅ RAG - Get similar docs using ChromaDB
def get_similar_docs(query: str, k: int = 4):
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_model
    )
    return vectorstore.similarity_search(query, k=k) #

# ✅ RAG - Standalone run for external use
def run_rag(query: str):
    relevant_docs = get_similar_docs(query)
    if not relevant_docs:
        return "❌ No relevant documents found."

    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {query}
"""
    return get_llm_response(prompt)

# ✅ Class for chat with memory + RAG
class LlamaChat:
    def __init__(self, cacheid: str):
        self.cacheid = cacheid

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You're a helpful assistant. Use the context if relevant.\n\nContext:\n{context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        from langchain_core.runnables import RunnableLambda
        #This is just a normal Python function, not a LangChain component yet.
        def azure_llm_chain(input_dict):
            full_prompt = self.prompt.invoke(input_dict)
            return {"content": get_llm_response(full_prompt.to_string())}

        self.chain = RunnableLambda(azure_llm_chain) #this makes it a LangChain component
        from langchain_core.runnables import RunnableLambda

# def say_hello(name):
#     return f"Hello {name}!"

# runnable = RunnableLambda(say_hello)

# # Now you can do this:
# print(runnable.invoke("Bro"))  # Hello Bro!
    def get_redis_history(self) -> BaseChatMessageHistory:
        return RedisChatMessageHistory(
            session_id=self.cacheid,
            redis_url="redis://localhost:6379",
            ttl=3600
        )

    def chat(self, user_input: str):
        similar_docs = get_similar_docs(user_input)
        context = "\n".join([doc.page_content for doc in similar_docs]) if similar_docs else "No relevant context found."

        chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_redis_history,
            input_messages_key="input",
            history_messages_key="history"
        )

        result = chain_with_history.invoke(
            {"input": user_input, "context": context},
            config={"configurable": {"session_id": self.cacheid}}
        )

        return {
            "type": type(result).__name__,
            "message": result["content"]
        }

    def get_chat_history(self):
        history = self.get_redis_history()
        return [
            {"type": type(msg).__name__, "message": msg.content}
            for msg in history.messages
        ]

#working code before implementing embedding and rag
# class LlamaChat:
#     def __init__(self, cacheid):
#         self.cacheid = cacheid
#         self.llm = ChatOllama(model="llama3.1:8b", temperature=0.7)
#         self.prompt = ChatPromptTemplate.from_messages([
#             ("system", "You're a helpful assistant. Answer user questions clearly."),
#             MessagesPlaceholder(variable_name="history"),
#             ("human", "{input}")
#         ])
#         self.chain = self.prompt | self.llm

#     def get_redis_history(self) -> BaseChatMessageHistory:
#         return RedisChatMessageHistory(
#             session_id=self.cacheid,
#             redis_url="redis://localhost:6379",
#             ttl=3600  # 1 hour
#         )

#     def chat(self, user_input: str):
#         chain_with_history = RunnableWithMessageHistory(
#             self.chain,
#             self.get_redis_history,
#             input_messages_key="input",
#             history_messages_key="history"
#         )
#         result = chain_with_history.invoke(
#             {"input": user_input},
#             config={"configurable": {"session_id": self.cacheid}}
#         )
#         return {
#             "type": type(result).__name__,
#             "message": result.content
#         }

#     def get_chat_history(self):
#         history = self.get_redis_history()
#         return [
#             {"type": type(msg).__name__, "message": msg.content}
#             for msg in history.messages
#         ]


# The `choices` list in the OpenAI (and Azure OpenAI) API response contains **all the generated completions** for your request.

# - `choices[0]` is the **first** generated response (the default and most common case).
# - If you set the `n` parameter (e.g., `n=3`), the API will return multiple completions, and you can access them as `choices[0]`, `choices[1]`, `choices[2]`, etc.
# - By default, `n=1`, so only `choices[0]` is present.

# **Summary:**  
# - `choices[0]` is the first (and usually only) response.
# - If you request more completions, you can access them with higher indices (`choices[1]`, `choices[2]`, ...).  
# - Most code uses only `choices[0]`.

# Great question!  
# The two functions serve **different purposes** in your Retrieval-Augmented Generation (RAG) pipeline:

# ---

# ### 1. `get_similar_docs(...)`  
# This function **retrieves relevant document chunks** from your vector database (Chroma) based on the user's query.

# - It uses the embedding model to convert the query into a vector.
# - It searches for the most similar vectors (chunks) in your database.
# - It returns the top `k` relevant chunks (context).

# **Purpose:**  
# To find and provide context from your ingested documents that are relevant to the user's question.

# ---

# ### 2. `completion = azure_client.chat.completions.create(...)`  
# This line **generates a response** from the LLM (Azure OpenAI) using the prompt you provide.

# - The prompt often includes the context retrieved by `get_similar_docs(...)`.
# - The LLM uses this context to answer the user's question more accurately.

# **Purpose:**  
# To generate a natural language answer, possibly using the context found by `get_similar_docs(...)`.

# ---

# ### **How they work together (RAG):**

# 1. **Retrieve:**  
#    Use `get_similar_docs(query)` to fetch relevant document chunks.

# 2. **Augment:**  
#    Add those chunks as context to your prompt.

# 3. **Generate:**  
#    Send the prompt (with context) to the LLM using `completion = ...`.

# ---

# **Summary:**  
# - `get_similar_docs` = finds relevant info (context) from your data.
# - `completion = ...` = generates the final answer using that context.

# Both are needed for a RAG system!
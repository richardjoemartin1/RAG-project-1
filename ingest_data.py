import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
load_dotenv()  

CHROMA_PERSIST_DIR = "./chroma_store"
urls = [
    "https://www.honolulu.gov/visitor-information/"
    # "https://www.zco.com/services/",
    # "https://www.zco.com/about/",
    # "https://www.zco.com/faq/",
    # "https://www.zco.com/",
    # "https://www.zco.com/animation-services/",
    # "https://www.zco.com/blog/",
    # "https://www.zco.com/mobile-app-development/",
    # "https://www.zco.com/ios-development-services/",
    # "https://www.zco.com/android-development-services/"
]

def ingest_website_content():
    docs = []

    for url in urls:
        try:
            loader = WebBaseLoader(url, requests_kwargs={"headers": {"User-Agent": os.getenv("USER_AGENT")}})
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            print(f"✅ Loaded: {url}")
        except Exception as e:
            print(f"❌ Failed to load {url}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_model
    )

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    # print("------------------- Ingesting Data ------------------")
    # print(f"🧩 Created texts /n {texts} ")
    # print("-----------------------------------------------------")
    # print(f"🧩 Created metadatas /n {metadatas} ")
    for i in range(len(texts) - 1):
        print(f"Chunk {i} end: {texts[i][-200:]}")
        print(f"Chunk {i+1} start: {texts[i+1][:200]}")
        print("---")
    vectorstore.add_texts(texts=texts, metadatas=metadatas)

    print(f"🧠 Added {len(texts)} chunks to Chroma DB.")

if __name__ == "__main__":
    ingest_website_content()
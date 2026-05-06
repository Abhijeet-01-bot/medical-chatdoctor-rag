from dotenv import load_dotenv
load_dotenv()
 
from typing import Dict
import warnings
warnings.filterwarnings("ignore")
 
from langchain_community.document_loaders import JSONLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
 
from langchain_google_genai import ChatGoogleGenerativeAI
 
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
import time
# ---------------- PROMPT ---------------- #
PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""
You are a medical consultant with expertise in doctor-patient conversations.
Use ONLY the given medical chats.
 
Context:
{context}
 
Chat History:
{chat_history}
 
Question:
{question}
 
Answer:
"""
)
 
# ---------------- MEMORY STORE ---------------- #
store = {}
 
def get_memory(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
 
# ---------------- PIPELINE ---------------- #
def load_pipeline():
 
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
 
    vectorstore = FAISS.load_local(
        "Faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
 
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )
 
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        temperature=0.7,
        max_tokens=2048
    )
 
    def retrieve_context(x):
        docs = retriever.invoke(x["question"])
        return "\n\n".join(doc.page_content for doc in docs)
 
    def format_chat_history(messages):
        if not messages:
            return ""
        return "\n".join(
            f"{m.type.upper()}: {m.content}"
            for m in messages
        )
 
    rag_chain = (
        {
            "context": retrieve_context,
            "chat_history": lambda x: format_chat_history(x["chat_history"]),
            "question": lambda x: x["question"],
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
 
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_memory,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
 
    return conversational_chain, retriever
# ---------------- QUERY FUNCTION ---------------- #
def ask_question(question: str, session_id: str = "default") -> Dict:
    start=time.time()
    chain, retriever = load_pipeline()
    latency=time.time()-start
    answer = chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )
 
    docs = retriever.invoke(question)
    sources = [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "content": doc.page_content[:200]
        }
        for doc in docs
    ]
 
    return {"answer": answer, "sources": sources}
 

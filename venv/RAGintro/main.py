from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/"),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content","post-title","post-header")
        )
    )
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

docs_splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=docs_splits,
    embedding=OpenAIEmbeddings(),
    collection_name="lilianweng_agent"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)
#rag prompt
prompt = hub.pull("rlm/rag-prompt")

# Define a simple query to test the retriever
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
   for chunk in rag_chain.stream({"question": "What is Maximum inner product search?"}):
       print(chunk, end="", flush=True)

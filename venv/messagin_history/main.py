from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
    MessagesPlaceholder(variable_name="messages"),
])

chain = prompt | model

config = {"configurable": {"session_id": "abcd1234"}}
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history(config["configurable"]["session_id"])
)
    

if __name__ == "__main__":
    while True:
        user_input = input(">")
        response = with_message_history.invoke({
            "messages": [
                HumanMessage(content=user_input)
            ],
            "config": config
        })
        print(response)

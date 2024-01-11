import dotenv
dotenv.load_dotenv()
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

def defineModel():
    # LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a nice chatbot having a conversation with a human. \
                You must answer in the same language than the user's question"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
    return conversation

def main():

    st.title("Multilingual Langchain Chatbot")

    # Initialize the conversation object and chat history in session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = defineModel()
        st.session_state.chat_history = []

    user_input = st.text_input("Enter your question:")
    submitted = st.button("Submit")

    if submitted:
        if user_input.lower() == "end":
            st.warning("Conversation ended.")
        else:
            response = st.session_state.conversation({"question": user_input})
            st.session_state.chat_history.append({"user": user_input, "bot": response['text']})

            # Display chat history
            st.write("Chat History:")
            for entry in st.session_state.chat_history:
                st.write(f"User: {entry['user']}")
                st.write(f"Bot: {entry['bot']}")

if __name__ == "__main__":
    main()
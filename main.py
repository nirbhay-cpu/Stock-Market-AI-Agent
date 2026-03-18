# import libraries

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from agent import run_stock_agent

# Show Title on the streamlit app using HTML.

st.set_page_config(layout="centered", page_title="Stock Assistant")

st.html(
    """<div style="display:flex; flex-direction:column; align-items:center; margin-top:40px; margin-bottom:30px;">

        <h1 style="font-size:70px; margin-bottom:10px; text-align:center;">
            Stock Assistant
        </h1>

        <p style="font-size:18px; max-width:900px; text-align:center; color:#9aa0a6;">
            Real-time stock prices • Latest market news • Finance research papers • Terminologies
        </p>

    </div>""")


# func for formatting chat history so that it passes to agent
def format_chat_history(messages):
    history = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            history.append(f"Assistant: {msg.content}")

    return "\n".join(history)



# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)


# Chat input
user_input = st.chat_input("Ask about stocks...")

if user_input:

    # Show user message on the right
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Get response from LLM / agent
    with st.spinner("Analyzing stock data..."):

        try:

            chat_history = format_chat_history(st.session_state.chat_history)
            response = run_stock_agent(user_input, chat_history)

        except ValueError as e:
            response = "I couldn't process the response. Please try asking differently."

    # Show assistant response on the left
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save assistant message
    st.session_state.chat_history.append(AIMessage(content=response))



#  --- END ---





# for logging the file , run this command in terminal -
# 
#                   command ->  python -u -m streamlit run main.py | Tee-Object -FilePath my_agent_logs.txt


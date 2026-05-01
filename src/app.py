import streamlit as st
import requests

st.set_page_config(page_title="AI Agent System", layout="centered")

st.title("AI Agent System (RAG + SQL + Agent)")

user_input = st.text_input("Ask anything:")

col1, col2 = st.columns([1, 1])
submit = col1.button("Submit")
clear = col2.button("Clear")

if clear:
    st.rerun()

API_URL = "https://amogh781-ai-agent.hf.space/query"

if submit:
    if user_input:
        st.chat_message("user").write(user_input)

        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    API_URL,
                    json={"question": user_input},
                    timeout=30
                )

                if response.status_code == 200:
                    answer = response.json().get("response", "No response")
                else:
                    answer = "API Error: Unable to get response"

        except Exception:
            answer = "Something went wrong. Please try again."

        st.chat_message("assistant").write(answer)

    else:
        st.warning("Please enter a question.")

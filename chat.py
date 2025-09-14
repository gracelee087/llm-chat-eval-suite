import streamlit as st


from dotenv import load_dotenv


from llm import get_ai_response

st.set_page_config(page_title="Unity Financial Group Chatbot", page_icon="🤖")

st.title("🤖 Financial Reporting Standards & Employee Handbook Chatbot")
st.caption("I can answer any questions you have about Financial Reporting Standards and the Employee Handbook!")

load_dotenv()

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])




if user_question := st.chat_input(placeholder="Please tell me what you'd like to know!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("Generating an answer."):
        ai_response = get_ai_response(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})

import streamlit as st
import langchain_helper as lch

st.title("Benny's Youtube Assistant")


with st.sidebar:
    with st.form(key='my form'):
        youtube_url = st.sidebar.text_area(
            label ="What is the URL of the Youtube video?",
            max_chars=100
        )

        query = st.sidebar.text_area(
            label = "What would you like to know about the video?",
            key="query"
        )
        openai_api_key = st.sidebar.text_input(
            label="OpenAI API Key",
            key= "langchain_search_api_key_openai",
            type="password"
        )
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

        submit_button = st.form_submit_button(label='Submit')


if query and youtube_url:
    if not openai_api_key:
        st.info("Please provide your OpenAI API key to continue.")
        st.stop()
    else:
        db = lch.create_db_from_youtube_video_url(youtube_url)
        response = lch.get_response_from_query(db, query)
        st.subheader("Answer:")
        st.write(response)
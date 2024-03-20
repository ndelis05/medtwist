import streamlit as st
import tempfile
import requests
import json
import base64
from audio_recorder_streamlit import audio_recorder
import openai
from openai import OpenAI

import time

openai.api_key = st.secrets["password"]
client = OpenAI()

def transcribe_audio(audio_file_path):
    audio_file = open(audio_file_path, "rb")
    transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file, 
    response_format="text"
    )
    return transcript
    

@st.cache_data
def patient(doctor_input):
    # Send a message to the model asking it to summarize the text
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You're a patient with appendicitis. You're in the ER."},
            {"role": "user", "content": f' Doctor question: {doctor_input}',},
        ],
    )
    # Return the content of the models response
    return response.choices[0].message.content



def any_llm(client, headers, model_settings, messages):
    response = client.chat.completions.create(
        model=model_settings["model"],
        headers=headers,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0.0,
        stop=["\n", " Human:", " AI:"],
        n=1,
        stream=False,
        logprobs=None,
        echo=False,
        stop_sequence=None,
        best_of=1,
        logit_bias=None,
        return_prompt=False,
        return_metadata=False,
        return_prompt_prefix=False,
        expand=False,
        **messages,
    )
    return response.choices[0].text
    


st.set_page_config(page_title="Patient Simulations for Education", page_icon="📖")
st.title("📖 Patient Simulations for Education")
with st.expander("About"):
    st.write("This is a collection of patient simulations for education. It is not intended for use by patients or the general public.")
    st.write("This is a work in progress. Please contact David Liebovitz, MD if you have any questions or feedback.")


with st.sidebar:

        
    audio_bytes = audio_recorder(
    text="Click. WAIT 3, 2, 1... START!",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_name="microphone",
    icon_size="4x",
    )
if audio_bytes:
    # Save audio bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        fp.write(audio_bytes)
        audio_file_path = fp.name



        doctor_words = transcribe_audio(audio_file_path)
        st.chat_message("doctor").write(doctor_words)
        response = patient(doctor_words)
        st.chat_message("assistant").write(response)

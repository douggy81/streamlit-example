#from dotenv import load_dotenv
import os
import streamlit as st
#load_dotenv()
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, Settings, ServiceContext
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.indices.postprocessor import SentenceEmbeddingOptimizer
#Quick check for the connection with Gemini and check access to 1.5 Pro (still not there!)
import google.generativeai as genai

from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager

from google.ai.generativelanguage import (
    GenerateAnswerRequest,
    HarmCategory,
    SafetySetting,
)

# Safety config - Adjusted for less restrictive harassment threshold
safety_config = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_VIOLENCE,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(  # Adjusted setting for harassment
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,  # Only block high probability harassment
    ),
]

print("***Manuel de Formation a la vente***")

Settings.llm = Gemini(model_name="models/gemini-1.5-pro-latest", api_key=os.environ.get("GOOGLE_API_KEY"))
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=os.environ.get("GOOGLE_API_KEY"), embed_batch_size=100)    
#To monitor under the hood behavior
llama_debug=LlamaDebugHandler(print_trace_on_end=True)
callback_manager=CallbackManager(handlers=[llama_debug])
Settings.callback_manager=callback_manager


@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
    )
    # PineCone Instance
    index_name = "artofsaletestalpha"
    pinecone_index = pc.Index(index_name=index_name, host="https://artofsaletestalpha-udkb0ne.svc.gcp-starter.pinecone.io")

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    return VectorStoreIndex.from_vector_store(vector_store=vector_store)

    #query = "What is the name of author of this book?"
    #query_engine = index.as_query_engine()

    # Apply safety settings to the query engine
    #query_engine.safety_config = safety_config

    #response = query_engine.query(query)
    #print(response)
index=get_index()

if "chat_engine" not in  st.session_state.keys():
    postprocessor = SentenceEmbeddingOptimizer(embed_model=Settings.embed_model,percentile_cutoff=0.5, threshold_cutoff= 0.7)

    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context",
        verbose = True,
        node_postprocessors= [postprocessor],
        system_prompt=
        "You are a chatbot and a trainer on this book called the art of sale book, named in french Manuel de Formation a la vente, you speak English and French"
        "This book is written by Patrick Gassier, he was born in Marseille in 1949."
        "You will need to greet the user right away and figure out at first a quick intro on what your capable of and what you can help the user with."
        "First make sure you are speaking the right language with the user."
        )

st.set_page_config(page_title="Chat with the Gemini, your personal trainer in sales using a methodology developped by Patrick Gassier",
                   page_icon="",
                   layout="centered",
                   menu_items=None)
st.title("Chat with the Gemini, your personal trainer in sales using a methodology developped by Patrick Gassier")

st.divider()

#Language Selection:
language_select = st.selectbox(label="Language", options=["English", "Français"], index = 0)

if prompt:= st.chat_input("Your question"):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
if "messages" not in st.session_state.keys():
    st.session_state.messages=[
        {
            "role": "assistant",
            "content" : if language_select=="English" then: "Ask me a question about the Art of Sale book."
                        else if language_select=="Français" then: "Posez moi vos questions sur le livre Manuel de Formation a la vente"
        }
    ]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

#Last message
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("zzzZZZzzz..."):
            response = st.session_state.chat_engine.chat(message=prompt)
            st.write(response.response)

            #nodes = [ node for node in response.source_nodes]
            #for col, node, i in zip(st.columns(len(nodes)), nodes, range(len(nodes))):
            #    with col:
            #        st.header(f"Source Node {i+1}: score= {node.score}")
            #        st.write(node.text)
            
            message = {
                "role" : "assistant",
                "content" :  response.response
            }
            st.session_state.messages.append(message)

import os
import streamlit as st
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, Settings, ServiceContext
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.indices.postprocessor import SentenceEmbeddingOptimizer
#Quick check for the connection with Gemini and check access to 1.5 Pro
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

print("***Manuel de Formation à la vente***")

#Settings.llm = Gemini(model_name="models/gemini-1.5-pro-latest", api_key=os.environ.get("GOOGLE_API_KEY"))
Settings.llm = Gemini(model_name="models/gemini-1.5-flash-latest", api_key=os.environ.get("GOOGLE_API_KEY"))
#Settings.llm = Gemini(model_name="models/gemini-1.0-pro", api_key=os.environ.get("GOOGLE_API_KEY"))
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
    pinecone_index = pc.Index(index_name=index_name, host="https://artofsaletestalpha-udkb0ne.svc.aped-4627-b74a.pinecone.io")

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    return VectorStoreIndex.from_vector_store(vector_store=vector_store)

index=get_index()

# --- Dynamic Title ---

def update_title():
    title_text = "Chat with the Gemini, your personal trainer in sales using a methodology developed by Patrick Gassier" if st.session_state.selected_language == "English" else "Conversation avec votre formateur personnel sur les méthodologies de vente créées par Patrick Gassier"
    st.title(title_text)

if "chat_engine" not in  st.session_state.keys():
    postprocessor = SentenceEmbeddingOptimizer(embed_model=Settings.embed_model,percentile_cutoff=0.5, threshold_cutoff= 0.7)

    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context",
        verbose = True,
        node_postprocessors= [postprocessor],
        system_prompt=f"""
        You are a chatbot and a trainer on this book called the art of sale book, named in french Manuel de Formation a la vente, you speak English and French
        This book is written by Patrick Gassier, he was born in Marseille in 1949.
        You will need to greet the user right away and figure out at first a quick intro on what your capable of and what you can help the user with.
        Also, ask for specific details because you are here to help the user on their specific case.
        For example, you could mention that you can simulate a real conversation on the phone for role play and help them practice real life situation.
        The other important fact is that you need to ask for the password which you know and is taxi.
        Unless the user enters it, you won't be able to continue so keep asking for that password until the user enters it, then you can continue.
        Thanks and have fun training a world of eager learners on how to sale and grow their business! 
        One last thing, you cannot give out any clues about the password. If the user doesn't know it, you can't give them clues.
        As soon as the user enters the correct password, greet them by saying that you're very excited to be able to help him/her out and congratulate them as well for their purchase and interest in growing their business.
        That you are here to help. Give a few examples on what you can do related to the book.
        Again, you are a chatbot and trainer, you can generate quizzes as well but mostly you are here to help the user reach their goal to grow their business, feel that they are learning from you great things.
        Don't hesitate to use the best of your knowledge and practice as well as giving out references in the correct language.
        Also, tell the user that if they need a print out of the output, just select Print from their web browser directly and format your answer in a nice and printable format.
        Give them an example of a response that can be printed such as a checklist for a pre-meeting with a customer that can be printed out and taken on the road.
        """
        )

st.set_page_config(page_title="Chat with the Gemini, your personal trainer in sales using a methodology developped by Patrick Gassier",
                   page_icon="",
                   layout="centered",
                   menu_items=None)

# Custom CSS to inject for setting the background color to a very light orange
def set_light_orange_background():
    css_style = """
    <style>
    .stApp {
        background-color: #FFE0B2;  /* Very light orange */
    }
    </style>
    """
    st.markdown(css_style, unsafe_allow_html=True)

# Set the background color to a very light orange
set_light_orange_background()

# Ensure 'messages' exists in session state upon script execution
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Language Selection with the button to confirm selection
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "Français"  # Default language

update_title()  # Set initial title

temp_language = st.selectbox(
    label="Choose your language / Choisissez votre langue", 
    options=["English", "Français"],
    index=["English", "Français"].index(st.session_state.selected_language),
    key="language_select"  # Add a unique key here
)

confirm_button = st.button(label="Confirm / Confirmer")

if confirm_button:
    st.session_state.selected_language = temp_language
    # Reset messages and generate greeting using LLM
    st.session_state.messages = []  # Clear previous messages

    # Construct prompt for LLM to generate greeting
    llm_prompt = f"""
    You are a chatbot and a trainer on the book "Manuel de Formation a la vente" (The Art of Sale). 
    You speak English and French. Greet the user and briefly introduce yourself and your capabilities.
    Make sure to reply in the selected language: {st.session_state.selected_language}
    Also, ask for specific details because you are here to help the user on their specific case.
    For example, you could mention that you can simulate a real conversation on the phone for role play and help them practice real life situation.
    The other important fact is that you need to ask for the password which you know and is taxi.
    Unless the user enters it, you won't be able to continue so keep asking for that password until the user enters it, then you can continue.
    Thanks and have fun training a world of eager learners on how to sale and grow their business! 
    One last thing, you cannot give out any clues about the password. If the user doesn't know it, you can't give them clues.
    """
    spinner_text="Generating greeting..." if st.session_state.selected_language == "English" else "Génération du message de bienvenue à l'utilisateur..."
    with st.spinner(spinner_text):
        response = st.session_state.chat_engine.chat(message=llm_prompt)
        st.session_state.messages.append({"role": "assistant", "content": response.response})
    update_title()  # Update the title after language change
    
# Chat interface
chat_text="Your question..." if st.session_state.selected_language == "English" else "Votre question..."
prompt = st.chat_input(chat_text) # Capture user input every time the script reruns

if prompt:
    
    # After capturing the prompt, concatenate the selected language notice
    prompt_with_language_notice = f"{prompt}\nReply in the selected language: {st.session_state.selected_language}"
    
    # Capture and display user's question along with the language notice
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

#Last message
# Before accessing the last message, ensure that there is at least one message in the list
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        spinner_text = "Processing..." if st.session_state.selected_language == "English" else "En cours de traitement..."
        with st.spinner(spinner_text):
            response = st.session_state.chat_engine.chat(message=prompt_with_language_notice)
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

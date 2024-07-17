import os
import streamlit as st
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, Settings, ServiceContext
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.indices.postprocessor import SentenceEmbeddingOptimizer
# Quick check for the connection with Gemini and check access to 1.5 Pro
import google.generativeai as genai
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from google.ai.generativelanguage import (
    GenerateAnswerRequest,
    HarmCategory,
    SafetySetting,
)

# --- Configuration ---

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

# Define API keys and model names
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_MODEL = "models/gemini-1.5-pro-latest"
EMBEDDING_MODEL = "models/text-embedding-004"
PINECONE_INDEX_NAME = "artofsaletestalpha"
PINECONE_HOST = "https://artofsaletestalpha-udkb0ne.svc.aped-4627-b74a.pinecone.io"

# --- Fonctions ---

@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    """Initialise et retourne l'index VectorStore."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(index_name=PINECONE_INDEX_NAME, host=PINECONE_HOST)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)

def update_title():
    """Met à jour le titre de la page en fonction de la langue sélectionnée."""
    title_text = "Chat with the Gemini, your personal sales trainer" if st.session_state.selected_language == "English" else "Conversation avec votre formateur personnel en vente"
    st.title(title_text)

# --- Initialisation ---

# Configuration du LLM et du modèle d'embedding
Settings.llm = Gemini(model_name=GEMINI_MODEL, api_key=GOOGLE_API_KEY)
Settings.embed_model = GeminiEmbedding(model_name=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY, embed_batch_size=100)    

# Configuration de la sécurité et des callbacks
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[llama_debug])
Settings.callback_manager = callback_manager

# Initialisation de l'index et du moteur de chat
index = get_index()
postprocessor = SentenceEmbeddingOptimizer(embed_model=Settings.embed_model, percentile_cutoff=0, threshold_cutoff=0)

# --- Interface Streamlit ---

st.set_page_config(page_title="The Art of Selling - AI Companion",
                   page_icon="",
                   layout="centered",
                   menu_items=None)

# Couleur de fond orange clair
st.markdown("""
<style>
.stApp {
    background-color: #FFE0B2;  
}
</style>
""", unsafe_allow_html=True)

# Gestion de l'état de la session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context",
        verbose=True,
        node_postprocessors=[postprocessor],
        system_prompt=f"""
        You are a chatbot and a trainer on the book "The Art of Selling - The French Method". 
        You speak English and French. You were trained on the content of the book and can provide insights,  advice,  and examples.
        You will need to greet the user right away and ask for the password "taxi" before continuing. 
        Once the user enters the password, congratulate them on their purchase and offer specific examples of how you can help them.
        For example, you could mention role-playing scenarios,  quizzes, or help with specific sales situations.
        You can also generate checklists,  scripts,  and other helpful materials.
        Feel free to use emojis (like 👍 or 😀) in your responses to make them more engaging.
        Remember to always reply in the selected language.
        """
    )

# Sélection de la langue
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "Français"

update_title()

temp_language = st.selectbox(
    label="Choose your language / Choisissez votre langue", 
    options=["English", "Français"],
    index=["English", "Français"].index(st.session_state.selected_language),
    key="language_select"
)

confirm_button = st.button(label="Confirm / Confirmer")

if confirm_button:
    st.session_state.selected_language = temp_language
    st.session_state.messages = []
    
    # Prompt pour le message de bienvenue
    llm_prompt = f"""
    You are a chatbot and a trainer on the book "The Art of Selling - The French Method". 
    You speak English and French. Greet the user and briefly introduce yourself and your capabilities.
    Make sure to reply in the selected language: {st.session_state.selected_language}
    """
    
    spinner_text = "Generating greeting..." if st.session_state.selected_language == "English" else "Génération du message de bienvenue à l'utilisateur..."
    with st.spinner(spinner_text):
        response = st.session_state.chat_engine.chat(message=llm_prompt)
        st.session_state.messages.append({"role": "assistant", "content": response.response})

# Interface de chat
chat_text = "Your question..." if st.session_state.selected_language == "English" else "Votre question..."
prompt = st.chat_input(chat_text) 

if prompt:
    prompt_with_language_notice = f"{prompt}\nReply in the selected language: {st.session_state.selected_language}"
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        spinner_text = "Processing..." if st.session_state.selected_language == "English" else "En cours de traitement..."
        with st.spinner(spinner_text):
            response = st.session_state.chat_engine.chat(message=prompt_with_language_notice)
            raw_response = response.response

            # Gestion de la convention de sortie
            if "__TASK__" in raw_response:
                st.markdown("## Checklist") 
                tasks = raw_response.split("__TASK__")[1].strip().split('\n') 
                for i, task in enumerate(tasks):
                    st.checkbox(task.strip(), key=f"task_{i}") 
            else:
                st.write(raw_response)
            
            st.session_state.messages.append({"role": "assistant", "content": response.response})

# Affichage des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

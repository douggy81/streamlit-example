import os
import streamlit as st
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, Settings, ServiceContext
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.indices.postprocessor import SentenceEmbeddingOptimizer

# --- Configuration ---

# API keys and model names
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
    title_text = "Votre Coach IA en Vente" if st.session_state.selected_language == "Français" else "Your AI Sales Coach"
    st.title(title_text)

def generate_greeting():
    """Génère un message de bienvenue en fonction de la langue sélectionnée."""
    greeting = "Bonjour ! Je suis votre coach IA en vente, prêt à vous aider à maîtriser les techniques de ce livre.  Pour commencer, veuillez entrer le mot de passe : " if st.session_state.selected_language == "Français" else "Hello! I'm your AI sales coach, ready to help you master the techniques in this book. To get started, please enter the password: "
    return greeting

# --- Initialisation ---

# Configuration du LLM et du modèle d'embedding
Settings.llm = Gemini(model_name=GEMINI_MODEL, api_key=GOOGLE_API_KEY)
Settings.embed_model = GeminiEmbedding(model_name=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY, embed_batch_size=100)    

# Initialisation de l'index et du moteur de chat
index = get_index()
postprocessor = SentenceEmbeddingOptimizer(embed_model=Settings.embed_model, percentile_cutoff=0, threshold_cutoff=0)

# --- Interface Streamlit ---

st.set_page_config(page_title="L'Art de la Vente - Assistant IA",
                   page_icon="",
                   layout="centered",
                   menu_items=None)

# Couleur de fond
st.markdown("""
<style>
.stApp {
    background-color: #FFE0B2;  
}
</style>
""", unsafe_allow_html=True)

# Gestion de l'état de la session

# Initialisation de la langue
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "Français"

# Initialisation des messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": generate_greeting()}]

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context",
        verbose=True,
        node_postprocessors=[postprocessor],
        system_prompt=f"""
        Vous êtes un chatbot formateur en vente, basé sur le livre "L'Art de la Vente - Une méthode à la française". 
        Vous parlez français et anglais. Vous avez été entraîné sur le contenu du livre et pouvez fournir des informations, des conseils et des exemples.
        Vous devez d'abord demander à l'utilisateur le mot de passe "taxi" avant de continuer.
        Une fois que l'utilisateur a entré le mot de passe, félicitez-le pour son achat et proposez des exemples précis de la façon dont vous pouvez l'aider.
        Par exemple, vous pouvez mentionner des scénarios de jeux de rôle, des quiz ou une aide pour des situations de vente spécifiques.
        Vous pouvez également générer des listes de contrôle, des scripts et d'autres documents utiles.
        N'hésitez pas à utiliser des emojis (comme 👍 ou 😀) dans vos réponses pour les rendre plus attrayantes.
        N'oubliez pas de toujours répondre dans la langue sélectionnée.
        """
    )

# --- Affichage du titre une seule fois ---
update_title()

# Sélection de la langue
temp_language = st.selectbox(
    label="Choose your language / Choisissez votre langue", 
    options=["English", "Français"],
    index=["English", "Français"].index(st.session_state.selected_language),
    key="language_select"
)

confirm_button = st.button(label="Confirm / Confirmer")

if confirm_button:
    st.session_state.selected_language = temp_language
    #st.session_state.messages = [] 
    st.session_state.messages = [{"role": "assistant", "content": generate_greeting()}]

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

## Affichage des messages
#for message in st.session_state.messages:
#    with st.chat_message(message["role"]):
#        st.write(message["content"])

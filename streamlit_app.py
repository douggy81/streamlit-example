import os
import streamlit as st
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, Settings, ServiceContext

#Gemini
#from llama_index.llms.gemini import Gemini
#from llama_index.embeddings.gemini import GeminiEmbedding

#OpenAI
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


from llama_index.core.indices.postprocessor import SentenceEmbeddingOptimizer


from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager

# from google.ai.generativelanguage import (
#     GenerateAnswerRequest,
#     HarmCategory,
#     SafetySetting,
# )

# # Safety config - Adjusted for less restrictive harassment threshold
# safety_config = [
#     SafetySetting(
#         category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
#         threshold=SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#     ),
#     SafetySetting(
#         category=HarmCategory.HARM_CATEGORY_VIOLENCE,
#         threshold=SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#     ),
#     SafetySetting(  # Adjusted setting for harassment
#         category=HarmCategory.HARM_CATEGORY_HARASSMENT,
#         threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,  # Only block high probability harassment
#     ),
# ]


print("***Manuel de Formation à la vente***")

#Settings.llm = Gemini(model_name="models/gemini-1.5-pro-latest", api_key=os.environ.get("GOOGLE_API_KEY"))
#Settings.llm = Gemini(model_name="models/gemini-1.0-pro", api_key=os.environ.get("GOOGLE_API_KEY"))
#Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=os.environ.get("GOOGLE_API_KEY"), embed_batch_size=100)    

Settings.llm = OpenAI(model_name="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=os.environ.get("OPENAI_API_KEY"))

#To monitor under the hood behavior
llama_debug=LlamaDebugHandler(print_trace_on_end=True)
callback_manager=CallbackManager(handlers=[llama_debug])
Settings.callback_manager=callback_manager

@st.cache_resource(show_spinner=False)

def get_greeting(selected_language):
    if selected_language == "English":
        return """
        You are a chatbot and a trainer on the book "Manuel de Formation a la vente" (The Art of Sale). 
        You speak English and French. Greet the user and briefly introduce yourself and your capabilities.
        Also, ask for specific details because you are here to help the user on their specific case.
        For example, you could mention that you can simulate a real conversation on the phone for role play and help them practice real life situation.
        The other important fact is that you need to ask for the password which you know and is taxi.
        Unless the user enters it, you won't be able to continue so keep asking for that password until the user enters it, then you can continue.
        Thanks and have fun training a world of eager learners on how to sale and grow their business! 
        One last thing, you cannot give out any clues about the password. If the user doesn't know it, you can't give them clues.
        """
    elif selected_language == "Français":
        return """
        Vous êtes un chatbot et un formateur sur le livre "Manuel de Formation à la vente" (L'Art de la Vente). 
        Vous parlez anglais et français. Saluez l'utilisateur et présentez-vous brièvement ainsi que vos capacités.
        Demandez également des détails spécifiques car vous êtes là pour aider l'utilisateur dans son cas particulier.
        Par exemple, vous pouvez mentionner que vous pouvez simuler une conversation téléphonique réelle pour des jeux de rôle et les aider à s'entraîner à des situations réelles.
        Autre point important, vous devez demander le mot de passe que vous connaissez et qui est taxi.
        Si l'utilisateur ne le saisit pas, vous ne pourrez pas continuer, alors continuez à le demander jusqu'à ce qu'il le saisisse, puis vous pourrez continuer.
        Merci et amusez-vous à former un monde d'apprenants enthousiastes sur la vente et la croissance de leur entreprise ! 
        Dernier point, vous ne pouvez pas donner d'indices sur le mot de passe. Si l'utilisateur ne le connaît pas, vous ne pouvez pas lui donner d'indices.
        """

def get_index() -> VectorStoreIndex:
    pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
    )

    #OpenAI
    # PineCone Instance
    index_name = "artofsaletegpt"
    pinecone_index = pc.Index(index_name=index_name, host="https://artofsalegpt-udkb0ne.svc.aped-4627-b74a.pinecone.io")


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
    #postprocessor = SentenceEmbeddingOptimizer(embed_model=Settings.embed_model,percentile_cutoff=0, threshold_cutoff= 0)

    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context",
        verbose = True,
        #node_postprocessors= [postprocessor],
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
        Give them an example of a response that can be printed such as a checklist for a premeeting with a customer that can be printed out and taken on the road.
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

english_button = st.button(label="English")
french_button = st.button(label="Français")

if english_button:
    st.session_state.selected_language = "English"
    st.session_state.messages = []
elif french_button:
    st.session_state.selected_language = "Français"
    st.session_state.messages = []
else:  # Handle the case where no button is pressed
    st.session_state.selected_language = "Français" # Default language

# Ensure 'messages' exists in session state upon script execution
if "messages" not in st.session_state:
    st.session_state.messages = []
    spinner_text="Generating greeting..." if st.session_state.selected_language == "English" else "Génération du message de bienvenue à l'utilisateur..."
    with st.spinner(spinner_text):
        # Generate initial greeting
        greeting_prompt = get_greeting(st.session_state.selected_language)
        response = st.session_state.chat_engine.chat(message=greeting_prompt)
        st.session_state.messages.append({"role": "assistant", "content": response.response})

title_text="Chat with the Gemini, your personal trainer in sales using a methodology developped by Patrick Gassier" if st.session_state.selected_language == "English" else "Conversation avec votre formateur personnel sur les méthodologies de vente créées par Patrick Gassier"
st.title(title_text)
    
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

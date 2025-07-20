import asyncio
import streamlit as st
from streamlit_theme import st_theme

from ConversationRepository import get_conversations_repository
from PaRAGraph import get_paragraph

BASE_URL = "http://www-01.ibm.com/support/docview.wss?uid="

def open_chat(chat_id):
    conversation_repository = get_conversations_repository()
    chat = conversation_repository.get_chat(chat_id)
    if chat:
        st.session_state["chat_id"] = chat_id
        st.session_state.messages = chat["chat_history"]
        st.session_state["retrieved_documents"] = chat["retrieved_documents"]

def save_chat():
    conversation_repository = get_conversations_repository()
    conversation_repository.save_chat(st.session_state["chat_id"], st.session_state.messages, st.session_state["retrieved_documents"])
    
def save_new_chat():
    conversation_repository = get_conversations_repository()
    new_chat_id = conversation_repository.save_new_chat(st.session_state.messages, st.session_state["retrieved_documents"])
    st.session_state["chat_id"] = new_chat_id
    
def create_new_chat():    
    st.session_state.messages = []
    st.session_state["retrieved_documents"] = []

def delete_chat(chat_id):
    conversation_repository = get_conversations_repository()
    conversation_repository.delete_chat(chat_id)
    
    st.session_state["retrieved_documents"] = []
    st.session_state.messages = []
    del st.session_state["chat_id"]
  
    

async def run(conversations_info):
    paRAGraph = get_paragraph()

    main_container = st.container()
    with main_container:
        chat_column, doc_column = st.columns(spec=[0.7,0.3], gap="medium", border=True)
        # Display document column
        with doc_column:
            content = ""
            for document in st.session_state["retrieved_documents"]:
                document_url = BASE_URL + document["document_id"]
                content += f"""<div class="sticky-documents">
                        <h4>{document["document_title"]}</h4>
                        <p>
                            {document["document_text"]}
                        </p>
                        <a href="{document_url}" target="_blank" class="document-link">
                            📄 Official Document Link
                        </a>
                    </div>"""
                
            html = f"""<div class="document_col">
                    <div class="sticky-box">
                        <h3 style="margin-top: 0;">Reference Documents</h3>{content}</div></div>"""
             
            st.markdown(html, unsafe_allow_html=True)

        # Display chat column
        with chat_column:
            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                if(message["role"] == "assistant"):
                    avatar = "images/granite_logo_white_bg.png"
                elif (message["role"] == "user"):
                    avatar ="🦖"
                else:
                    avatar = None

                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

        # Create new chat button
        st.sidebar.button(
            "New Chat", 
            type="secondary", 
            on_click=create_new_chat, 
            key = "New chat",
            icon=":material/chat_add_on:",
            use_container_width=True
        )

        st.sidebar.divider()
        
        col_chat_btns, col_chat_delete = st.sidebar.columns([0.8, 0.2], gap="small")

        # Display conversations
        for chat in conversations_info:
            
            with col_chat_btns:
                if chat["_id"] == st.session_state["chat_id"]:
                    btn_type = "secondary"
                else: 
                    btn_type = "tertiary"
                
                col_chat_btns.button(
                    chat["title"], 
                    type=btn_type, 
                    on_click=open_chat, 
                    key = "open-"+str(chat["_id"]), 
                    args = (chat["_id"],)
                )

            with col_chat_delete:
                col_chat_delete.button(
                    "", 
                    type="tertiary", 
                    on_click=delete_chat, 
                    key = "delete-"+str(chat["_id"]), 
                    args = (chat["_id"],),
                    icon = ":material/delete:"
                )   

        # React to user input
        if prompt := st.chat_input("Feel free to ask anything"):
            with chat_column:    
                # Display new messages            
                st.chat_message("user", avatar="🦖").markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                new_query = paRAGraph.rewrite_query(prompt)

                # Retrieve relevant docs
                with st.spinner("Generating the answer...", show_time=False, width="stretch"):
                    new_documents = await paRAGraph.retrieve_documents(new_query)
                    dict_translation = {ord("<"): "&lt;", ord(">"): "&gt;"}

                    translated_documents = []
                    for document in new_documents:
                        new_doc = document
                        new_doc["document_text"] = document["document_text"].translate(dict_translation)
                        translated_documents.append(new_doc)

                    # Add new documents to all retrieved documents
                    st.session_state["retrieved_documents"].extend(translated_documents)

                    # Generate and display system's answer
                    response = paRAGraph.generate_answer(prompt, new_documents)
                    st.chat_message("assistant", avatar="images/granite_logo_white_bg.png").markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # Save chat history
                    if len(st.session_state.messages) == 2:
                        save_new_chat()
                    else:
                        save_chat()
                    st.rerun()
                


if __name__ == "__main__":
    conversation_repository = get_conversations_repository()

    st.set_page_config(
        page_title="PaRAGraph",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load css style based on current theme
    with open("./style.css") as f:
        css = f.read()

    theme = st_theme()
    if theme and theme["base"] == "dark":
        css = css + """.sticky-documents a:hover {
                        background-color: #262730;
                        padding-left: 0.5rem;
                    })"""
    elif theme:
        css = css + """.sticky-documents a:hover {
                        background-color: #BFC5D3;
                        padding-left: 0.5rem;
                    })"""
        
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    db = conversation_repository.get_all()
    
    if "chat_id" not in st.session_state:
        # Load last modified chat
        if len(db) != 0:
            st.session_state.messages = db[0]["chat_history"]
            st.session_state["chat_id"] = db[0]["_id"]
            st.session_state["retrieved_documents"] = db[0]["retrieved_documents"]
        else:
            # No chats found in DB
            st.session_state.messages = []
            st.session_state["retrieved_documents"] = []
    
    conversations_info = []
    for chat in db:
        del chat["chat_history"]
        conversations_info.append(chat)
    
    st.title("📝 PaRAGraph")

    asyncio.run(run(conversations_info))
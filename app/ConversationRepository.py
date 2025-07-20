from datetime import datetime
from pymongo import MongoClient, DESCENDING
import streamlit as st

# Global cached instance
@st.cache_resource
def get_conversations_repository():
    """
    This function will be called only once across all reruns and sessions.
    The ConversationRepository instance will be cached and reused.
    """
    print("Creating new ConversationRepository instance")
    return ConversationRepository()


class ConversationRepository:

    def __init__(self):
        self.setup()
    
    def setup(self):
        client = MongoClient('mongodb://localhost:27017/')
        self.db = client.get_database("local")["conversations"]
        
    def get_chat(self, chat_id):
        chat = self.db.find_one({"_id": chat_id})
        return chat
    
    def get_all(self):
        return self.db.find({}).sort("last_modified", DESCENDING).to_list()
    
    def save_chat(self, chat_id, chat_history, retrieved_documents):
        new_values = {
                        "$set": {
                                    "last_modified": datetime.today().replace(microsecond=0),
                                    "chat_history": chat_history,
                                    "retrieved_documents": retrieved_documents
                                }
                    }
        self.db.update_one({"_id": chat_id}, new_values)
       
        
    def save_new_chat(self, chat_history, retrieved_documents):
    
        new_chat = {}
        new_chat["last_modified"] = datetime.today().replace(microsecond=0)
        new_chat["chat_history"] = chat_history
        
        # Truncate titles that exceed char limit
        if len(chat_history[0]["content"]) > 30:
            new_chat["title"] = chat_history[0]["content"][:27] + "..."
        else:
            new_chat["title"] = chat_history[0]["content"]
        new_chat["retrieved_documents"] = retrieved_documents
        
        inserted_chat = self.db.insert_one(new_chat)
        return inserted_chat.inserted_id

    def delete_chat(self, chat_id):
        self.db.delete_one({"_id": chat_id})
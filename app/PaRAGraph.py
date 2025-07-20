from elasticsearch import AsyncElasticsearch
import gc
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import NodeRelationship
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
import torch
import torch_directml
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration

# Global cached instance
@st.cache_resource
def get_paragraph():
    """
    This function will be called only once across all reruns and sessions.
    The PaRAGraph instance will be cached and reused.
    """
    print("Creating new PaRAGraph instance")
    return PaRAGraph()

class PaRAGraph:

    def __init__(self):
        self._retrieve_score_threshold = 0.95
        self._retrive_similarity_top_k = 1
        self.setup()
    

    def setup(self):
        # Set up gpu
        self.device = torch_directml.device()
        print(self.device)
        # Load embedding model for elastic search vector store index
        embed_model_name = "ibm-granite/granite-embedding-30m-english"
        embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name,
            device=self.device,
            normalize=True
        )
        embed_model._model = embed_model._model.to(self.device)
        print(f"Modello spostato su {self.device}")
        Settings.embed_model = embed_model


        async_client = AsyncElasticsearch(
            hosts=["http://localhost:9200"],
            node_class='httpxasync'
        )   

        # Open ElasticSearch DB
        elastic_search_store = ElasticsearchStore(
            index_name="techqa-index",
            es_url="http://localhost:9200",
            show_progress=True,
            es_client=async_client
        )

        self.vector_store_index = VectorStoreIndex.from_vector_store(elastic_search_store)

        self.retriever = self.vector_store_index.as_retriever(similarity_top_k=self._retrive_similarity_top_k)
        
        # Load Granite for answer generation
        LLM_name = "ibm-granite/granite-3.2-2b-instruct"
        self.LLM_tokenizer = AutoTokenizer.from_pretrained(LLM_name)
        self.LLM_model = AutoModelForCausalLM.from_pretrained(LLM_name).to(self.device)
        
        # Load query rewriter model
        query_rewriter_name = "catyung/t5l-turbo-hotpot-0331"
        self.query_rewriter_model = T5ForConditionalGeneration.from_pretrained(query_rewriter_name)
        self.query_rewriter_tokenizer = T5Tokenizer.from_pretrained(query_rewriter_name)
        

    def rewrite_query(self, query):
        rewrite_prompt = f"""rewrite a better search query: {query}
        answer:"""
        
        input_ids = self.query_rewriter_tokenizer(rewrite_prompt, return_tensors="pt").input_ids
        outputs = self.query_rewriter_model.generate(input_ids, max_new_tokens=100)
        new_query = self.query_rewriter_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return new_query
        


    async def retrieve_documents(self, query):
        # Retrieve relevant sections
        result_sections = await self.retriever.aretrieve(query)
        
        question_documents = []
        found_ids = []
        for section in result_sections:
            # Retrieve all sections with a score greater than threshold
            if section.score and section.score > self._retrieve_score_threshold:
                doc_id = section.metadata["document_id"] 
                doc_title = section.metadata["document_title"] 
                # Rebuild the original document
                if doc_id not in found_ids:
                    found_ids.append(doc_id)
                    first_node = section.node

                    # Get the first node (section) of the document
                    while NodeRelationship.PREVIOUS in first_node.relationships:
                        previous_id = first_node.relationships[NodeRelationship.PREVIOUS].node_id
                        first_node = self.vector_store_index.vector_store.get_nodes([previous_id])[0]
                        
                    cur_node = first_node
                    document = {}
                    document["document_id"] = doc_id
                    document["document_title"] = doc_title
                    document_text = ""
                
                    # Iterate all the section until the document is completely rebuilt
                    while NodeRelationship.NEXT in cur_node.relationships:
                        document_text +=  cur_node.text  
                        next_id = cur_node.relationships[NodeRelationship.NEXT].node_id
                        cur_node = self.vector_store_index.vector_store.get_nodes([next_id])[0]

                    # Get the last node (section) of the document
                    document_text +=  cur_node.text
                    document["document_text"] = document_text
                    question_documents.append(document)   
                    
        return question_documents
            
        
    def generate_answer(self, query, retrieved_documents):        
        # Prompt message for LLM
        prompt = """
            You are paRAGraph, a helpful RAG (Retrieval-Augmented Generation) system designed to answer user queries based only on the content of the retrieved documents.

            ### Key Instructions:
            1. **Only Use Retrieved Documents for Answers**: Provide an answer using specific, direct information in the retrieved documents. 
            2. **No Speculation**: Do not try to make inferences or use general knowledge.
            3. **Do Not Mention Documents**: Never refer to, mention, or include any details about the documents in your response. Do not say things like "According to the document," or "The document indicates...". Simply provide the answer.
            4. **No Extra Information**: Do not elaborate or provide additional context.
        """

        document_texts = []
        for document in retrieved_documents:
            document_texts.append(document["document_text"])
        
        # Create chat_template for the LLM
        messages = [
            {"role": "system", "content" : prompt},
            {"role": "user", "content": query},
            {"role": "system", "content": "Retrieved Documents:" + str(document_texts)}
        ]
        
        # Generate the answer 
        input_ids = self.LLM_tokenizer.apply_chat_template(
                                conversation = messages, 
                                return_tensors="pt",
                                thinking=False,
                                return_dict=True,
                                truncation = True,
                                max_length = (8192+4096),
                                controls = {"length":"short"},
                                add_generation_prompt=True).to(self.device)
        
        with torch.no_grad():
            output = self.LLM_model.generate(
                **input_ids,
                max_new_tokens=3072,
                temperature = 0.1,
                do_sample = True,
            )

        # Get the text of the generated answer
        response = self.LLM_tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Free memory
        torch.cuda.empty_cache()
        gc.collect()

        return response
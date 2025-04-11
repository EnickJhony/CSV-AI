import numpy as np
import pandas as pd
import difflib
import requests
from typing import List
from pathlib import Path
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
import requests
import json


# Load the CSV file
path_file = Path("C:/Users/esantos/Desktop/tbemail.csv")

# loader = CSVLoader(path_file)
df = pd.read_csv(path_file)

docs = []
for _, row in df.iterrows():
    doc_text = (
        f"id: {row['nmmetodo']}\n"
        f"Assunto: {row['nmsubject']}\n"
        f"Servidor de envio: {row['nrserveremail']}\n"
        f"Agendamento (cron): {row['txschedule']}\n"
        f"Método: {row['nmmetodo']}\n"
        f"Ativo: {row['bolativo']}\n"
        f"Última execução: {row['tmultimaexecucao']}"
    )
    docs.append(doc_text)

embeddings_model_path = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=embeddings_model_path,
)

embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

docs_with_id = [Document(page_content=doc, metadata={"source": i}) for i, doc in enumerate(docs)]

# deleta as informações do banco
# vector_store.delete_collection()

if vector_store._collection.count() == 0:
    # Adiciona os documentos ao banco de dados
    vector_store.add_documents(docs_with_id)

# quantidade de registros no banco
# print("Quantidade de itens do banco: " + str(vector_store._collection.count()))

query = "Qual o nome do id 30?"
results = vector_store.similarity_search(query, k=vector_store._collection.count())

# Mostra o resultado mais próximo
# print("Resultado mais proximo: " + results[0].page_content)

# Mostra todos os resultados
# for r in results:
#     print("---- Resultado ----")
#     print(r.page_content)

retriever = vector_store.as_retriever()

similar_chunks = retriever.invoke(query)

similar_texts = [ chunk.page_content for chunk in similar_chunks ]

sources = list(set([ chunk.metadata['source'] for chunk in similar_chunks ]))

prompt = "Considerando os textos abaixo:\n{}\n\nResponda a pergunta: {}".format(
    "\n".join(similar_texts), query
)


# URL do servidor do Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"

def conversar_com_ollama(model: str, mensagem: str):
    """
    Envia uma mensagem para o Ollama e retorna a resposta completa após o streaming.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": mensagem}]
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        response.raise_for_status()  # Levanta exceções para erros HTTP
        
        # Processa a resposta em partes
        resposta_completa = ""
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            try:
                # Cada chunk é um JSON separado
                chunk_data = json.loads(chunk)
                if "message" in chunk_data and "content" in chunk_data["message"]:
                    resposta_completa += chunk_data["message"]["content"]
            except json.JSONDecodeError:
                # Ignora chunks inválidos
                continue

        return resposta_completa
    except requests.exceptions.ConnectionError:
        return "Erro: Não foi possível conectar ao servidor Ollama."
    except requests.exceptions.Timeout:
        return "Erro: O tempo limite da solicitação foi excedido."
    except requests.exceptions.RequestException as e:
        return f"Erro ao se comunicar com o Ollama: {e}"


# def conversar_com_ollama(model: str, mensagem: str):
#     """
#     Envia uma mensagem para o Ollama e retorna a resposta.
    
#     :param model: Nome do modelo no Ollama (ex: "llama2", "gpt4all").
#     :param mensagem: Mensagem a ser enviada.
#     :return: Resposta do Ollama.
#     """
#     payload = {
#         "model": model,
#         "messages": [{"role": "user", "content": mensagem}]
#     }
    
#     try:
#         response = requests.post(OLLAMA_URL, json=payload)
#         response.raise_for_status()  # Levanta exceções para erros HTTP
#         return response.json().get("content", "Sem resposta do Ollama.")
#     except requests.exceptions.RequestException as e:
#         return f"Erro ao se comunicar com o Ollama: {e}"

# Exemplo de uso
# modelo = "granite3.2:2b"  # Substitua pelo modelo que você configurou no Ollama
# mensagem = "Olá, como você está?"


# Exemplo de uso
modelo = "granite3.2:2b"
mensagem = prompt
resposta = conversar_com_ollama(modelo, mensagem)
print("Prompt enviado: " + prompt)
print("A pergunta foi: " + query)
print("Resposta do Ollama:", resposta)
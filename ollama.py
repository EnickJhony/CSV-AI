import requests
import json

# URL do servidor do Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"

def conversar_com_ollama(mensagem: str):
    """
    Envia uma mensagem para o Ollama e retorna a resposta completa após o streaming.
    """
    payload = {
        "model": "gemma3:1b",
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

# Exemplo de uso
# modelo = "granite3.2:2b"
# mensagem = "Olá, como você está?"
# resposta = conversar_com_ollama(mensagem)
# print("Resposta do Ollama:", resposta)
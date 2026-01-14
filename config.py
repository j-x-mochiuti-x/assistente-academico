"""
Configurações centralizadas do projeto.
Facilita manutenção e evita valores hardcoded.
"""
import os
from pathlib import Path

# Diretórios do projeto
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Criar diretórios se não existirem
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# Configurações do Modelo LLM
LLM_CONFIG = {
    "model": "llama-3.3-70b-versatile",  # Modelo mais capaz para análise acadêmica
    "temperature": 0.3,  # Um pouco mais criativo que 0.2 para síntese
    "max_tokens": 2048   # Mais tokens para respostas elaboradas
}

# Configurações de Embeddings
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",  # Modelo leve e eficiente
    # Alternativa mais precisa (mas mais lenta): "sentence-transformers/msmarco-bert-base-dot-v5"
}

# Configurações de Chunking (divisão de texto)
CHUNK_CONFIG = {
    "chunk_size": 1200,      # Chunks maiores para contexto acadêmico
    "chunk_overlap": 200,    # Sobreposição para não perder contexto
}

# Configurações de Retrieval (busca)
RETRIEVAL_CONFIG = {
    "k": 5,  # Número de chunks relevantes a buscar
}

# Configurações da Interface
UI_CONFIG = {
    "page_title": "Assistente Acadêmico",
    "page_icon": "🎓",
    "layout": "wide"
}

# Configurações do Sistema RAG
RAG_SYSTEM_PROMPT = """Você é um assistente acadêmico especializado em análise de papers científicos.

Sua tarefa é responder perguntas baseando-se ESTRITAMENTE no contexto fornecido dos papers.

Diretrizes:
1. **Cite as fontes**: Sempre mencione de qual paper veio cada informação
2. **Seja preciso**: Se a resposta não estiver no contexto, diga claramente
3. **Estruture bem**: Use seções quando apropriado
4. **Compare quando pedido**: Faça comparações diretas entre estudos
5. **Linguagem acadêmica**: Use terminologia técnica, mas seja claro
"""
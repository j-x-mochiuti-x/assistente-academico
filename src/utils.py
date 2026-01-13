"""
Funções utilitárias do projeto.
"""
from typing import List
from langchain_core.documents import Document


def format_document_for_display(doc: Document, max_length: int = 200) -> str:
    """
    Formata um documento para exibição amigável.
    
    Args:
        doc: Documento LangChain
        max_length: Tamanho máximo do preview
        
    Returns:
        String formatada para exibição
    """
    content_preview = doc.page_content[:max_length]
    if len(doc.page_content) > max_length:
        content_preview += "..."
    
    metadata = doc.metadata
    page = metadata.get("page", "?")
    source = metadata.get("source_file", "desconhecido")
    
    return f"**[{source} - p.{page}]**\n{content_preview}"


def calculate_total_tokens(documents: List[Document]) -> int:
    """
    Estima o número total de tokens nos documentos.
    
    ⚠️ AVISO DE INCERTEZA: Esta é uma aproximação grosseira.
    Tokens reais dependem do tokenizador específico do modelo.
    Regra geral: ~4 caracteres = 1 token
    
    Args:
        documents: Lista de documentos
        
    Returns:
        Estimativa de tokens
    """
    total_chars = sum(len(doc.page_content) for doc in documents)
    estimated_tokens = total_chars // 4
    return estimated_tokens


def get_document_stats(documents: List[Document]) -> dict:
    """
    Calcula estatísticas sobre os documentos.
    
    Args:
        documents: Lista de documentos
        
    Returns:
        Dicionário com estatísticas
    """
    if not documents:
        return {
            "total_docs": 0,
            "total_chars": 0,
            "avg_doc_length": 0,
            "estimated_tokens": 0
        }
    
    total_chars = sum(len(doc.page_content) for doc in documents)
    
    return {
        "total_docs": len(documents),
        "total_chars": total_chars,
        "avg_doc_length": total_chars // len(documents),
        "estimated_tokens": calculate_total_tokens(documents)
    }
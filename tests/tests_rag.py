import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_engine import RAGEngine
from langchain_core.documents import Document

def test_rag_initialization():
    engine = RAGEngine()

    assert engine.embedding_model_name is not None
    assert engine.llm_model_name is not None
    print("✅ RAG inicializado com sucesso")

def test_document_formatting():
    engine = RAGEngine()

    test_docs = [
        Document(
            page_content="Este é um teste de conteúdo acadêmico sobre machine learning.",
            metadata={"source_file": "paper1.pdf", "page": 1, "chunk_index": 0}
        ),
        Document(page_content="Outro chunk sobre deep learning e redes neurais.",
            metadata={"source_file": "paper2.pdf", "page": 3, "chunk_index": 5}
        )
    ]

    formatted = engine.format_documents(test_docs)

    assert "paper1.pdf" in formatted
    assert "paper2.pdf" in formatted
    assert "machine learning" in formatted
    print("✅ Formatação de documentos funciona")
    print(f"\nExemplo formatado:\n{formatted[:200]}...")

if __name__ == "__main__":
    print("🧪 Executando testes do RAG...\n")
    test_rag_initialization()
    print()
    test_document_formatting()
    print("\n🎉 Todos os testes passaram!")
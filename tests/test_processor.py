"""
Testes para o módulo document_processor.
Execute: pytest tests/
"""
import sys
from pathlib import Path

# Add o diretório raiz no Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor

def test_text_cleaning():
    processor = DocumentProcessor()

    dirty_text = "Texto   com    múltiplos espaços\n\n\n\ne quebras\n\n\nexcessivas"
    clean_text = processor.clean_text(dirty_text)

    assert "   " not in clean_text
    assert "\n\n\n" not in clean_text
    print("✅ Teste de limpeza passou")

def test_chunk_creation():
    from langchain_core.documents import Document

    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

    #cria doc teste
    long_text = "Teste. " * 100
    test_doc = Document(page_content=long_text, metadata={"page": 1})

    chunks = processor.split_documents([test_doc])

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 150 for c in chunks)
    print(f"✅ Teste de chunking passou - {len(chunks)} chunks criados")


if __name__ == "__main__":
    test_text_cleaning()
    test_chunk_creation()
    print("\n🎉 Todos os testes passaram!")
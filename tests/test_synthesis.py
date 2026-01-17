"""
Testes para o módulo de síntese.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.synthesis import PaperSynthesizer
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import os


def test_synthesis_structure():
    """Testa se a classe inicializa corretamente."""
    # Precisa de API key para criar LLM
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️ GROQ_API_KEY não configurada, pulando teste")
        return
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    synthesizer = PaperSynthesizer(llm)
    
    assert synthesizer.llm is not None
    print("✅ PaperSynthesizer inicializado com sucesso")


def test_single_paper_summary():
    """Testa resumo de um único paper."""
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️ GROQ_API_KEY não configurada, pulando teste")
        return
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    synthesizer = PaperSynthesizer(llm)
    
    # Cria documento de teste
    test_docs = [
        Document(
            page_content="""
            Este estudo teve como objetivo investigar a prevalência de parasitas em cães.
            Metodologia: Foram coletadas 100 amostras de sangue de cães em abrigos.
            Utilizamos PCR para detecção molecular e testes sorológicos.
            Resultados: 30% dos cães apresentaram anticorpos positivos.
            Conclusão: Alta prevalência indica necessidade de medidas preventivas.
            """,
            metadata={"author": "Silva", "year": 2024, "source_file": "test.pdf"}
        )
    ]
    
    summary = synthesizer.summarize_single_paper(test_docs, focus="metodologia")
    
    assert summary["success"] == True
    assert len(summary["summary"]) > 0
    assert "metodologia" in summary["focus"].lower()
    
    print("✅ Resumo de paper individual funciona")
    print(f"\nExemplo de resumo:\n{summary['summary'][:200]}...")


if __name__ == "__main__":
    print("🧪 Executando testes de síntese...\n")
    test_synthesis_structure()
    print()
    test_single_paper_summary()
    print("\n🎉 Todos os testes de síntese passaram!")
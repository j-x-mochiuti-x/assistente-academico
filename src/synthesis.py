from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

class PaperSynthesizer:
    def __init__(self, llm: ChatGroq):
        self.llm = llm

    def summarize_single_paper(
            self,
            documents: List[Document],
            focus: str = "metodologia"
    ) -> Dict[str, Any]:
        
        #Combina todos os chunks do paper
        full_text ="\n\n".join([doc.page_content for doc in documents])

        # Pega metadados do primeiro chunk
        metadata = documents[0].metadata if documents else {}

        #Define prompt baseado no foco
        focus_prompts = {
            "metodologia": """Analise APENAS a metodologia deste paper:

            {text}

            Extraia e resuma:
            1. **Tipo de estudo**: (experimental, observacional, revisão, etc)
            2. **Amostra**: Tamanho e características
            3. **Técnicas/Métodos**: Principais abordagens usadas
            4. **Análise de dados**: Como os dados foram analisados

            Seja conciso (máx 150 palavras)""",
            "resultados":"""Analise APENAS os resultados deste paper:

{text}

Extraia:
1. **Principais achados**: Top 3 resultados mais importantes
2. **Dados quantitativos**: Percentuais, valores estatísticos
3. **Significância**: O que os resultados indicam

Seja conciso (máx 150 palavras)""",
            "limitacoes": """Anlise as limitações deste paper:

{text}

Identifique:
1. **Limitações metodológicas**: Problemas no método
2. **Limitações amostrais**: Problemas com a amostra
3. **Gaps de pesquisa**: O que falta investigar

Seja conciso (máx 100 palavras).""",
        "completo": """Faça um resumo executivo deste paper:

{text}

Estruture em:
1. **Objetivo**: Por que o estudo foi feito
2. **Metodologia**: Como foi feito (resumo)
3. **Resultados**: O que foi encontrado
4. **Conclusão**: Implicações principais

Seja conciso (máx 200 palavras)."""
        }

        prompt_template = focus_prompts.get(focus, focus_prompts["completo"])

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Vcê é um revisor acadêmico especializado."),
            ("human", prompt_template)
        ])

        #LIMITA TEXTO PARA NÃO ESTOURAR TOKENS!!!
        text_truncated = full_text[:8000]
        chain = prompt | self.llm

        try:
            response = chain.invoke({"text": text_truncated})
            sumary = response.content if hasattr(response, 'content') else str(response)

            return {
                "summary": sumary,
                "matadata": metadata,
                "focus": focus,
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "summary": None,
                "metadata": metadata,
                "focus": focus,
                "success": False,
                "error": str(e)
            }
    
    def compare_papers(
            self,
            summaries: List[Dict[str, Any]],
            comparision_focus: str = "metodologia"
    ) -> str:
        if not summaries:
            return "Nenhum resumo disponível para comparação."
        
        # Filtra apenas resumos bem-sucedidos
        valid_summaries = [s for s in summaries if s["success"]]

        if not valid_summaries:
            return "Nenum resumo válido para comparação."
        
        # Constrói texto com todos os resumos
        summaries_text = ""
        for i, summary in enumerate(valid_summaries, 1):
            meta = summary["metadata"]
            author = meta.get("author", "Autor desconhecido")
            year = meta.get("year", "Ano desconhecido")

            summaries_text += f"\n\n**Paper {i} - {author} ({year}):**\n{summary['summary']}"
        
        # Prompt de comparação
        comparison_prompt = f"""Você é um revisor de literatura acadêmica. Compare os seguintes {len(valid_summaries)} papers focando em {comparision_focus}:

{summaries_text}

Gere uma síntese comparativa estruturada:

## 📊 Comparação de {comparision_focus.title()}

### Semelhanças
- Liste aspectos comuns entre os estudos

### Diferenças
- Destaque abordagens distintas

### Padrões Identificados
- Tendências ou consensos emergentes

### Gaps de Pesquisa
- O que ainda precisa ser investigado

Seja técnico mas claro. Use bullets e seções.
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é um especialista em revisão sistemática de literatura."),
            ("human", comparison_prompt)
        ])
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({})
            return response.content if hasattr(response, 'content') else str(response)
        
        except Exception as e:
            return f"Erro ao gerar comparação: {str(e)}"
    
    def generate_literature_review(
        self,
        papers_documents: Dict[str, List[Document]],
        focus: str = "completo"
    ) -> Dict[str, Any]:
        
        print(f"📚 Iniciando revisão de literatura de {len(papers_documents)} papers...")
        
        # FASE MAP: Resume cada paper individualmente
        summaries = []
        for paper_name, documents in papers_documents.items():
            print(f"   ⏳ Analisando: {paper_name}...")
            summary = self.summarize_single_paper(documents, focus=focus)
            summaries.append(summary)
        
        print(f"   ✅ {len(summaries)} papers analisados")
        
        # FASE REDUCE: Compara todos os resumos
        print(f"   ⏳ Gerando síntese comparativa...")
        comparison = self.compare_papers(summaries, comparison_focus=focus)
        
        print(f"   ✅ Revisão de literatura concluída")
        
        return {
            "individual_summaries": summaries,
            "comparative_synthesis": comparison,
            "total_papers": len(papers_documents),
            "focus": focus
        }
    
    def quick_compare(llm: ChatGroq, papers_docs: Dict[str, List[Document]]) -> str:
    
        synthesizer = PaperSynthesizer(llm)
        result = synthesizer.generate_literature_review(papers_docs, focus="completo")
        return result["comparative_synthesis"]
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime


class PaperSynthesizer:
    """
    Sintetizador de papers acadêmicos usando Map-Reduce.
    
    Fluxo:
    1. MAP: Analisa cada paper individualmente
    2. REDUCE: Combina análises em síntese comparativa
    3. EXPORT: Formata resultado para diferentes saídas
    """
    
    def __init__(self, llm: ChatGroq):
       
        self.llm = llm
        self.analysis_cache = {} 
    
    def _truncate_text(self, text: str, max_chars: int = 8000) -> str:
        
        #Trunca texto para não estourar limite de tokens.
        if len(text) <= max_chars:
            return text
        
        # Tenta cortar em parágrafo completo
        truncated = text[:max_chars]
        last_paragraph = truncated.rfind('\n\n')
        
        if last_paragraph > max_chars * 0.8:  # Se encontrou parágrafo perto do fim
            return truncated[:last_paragraph] + "\n\n[...texto truncado...]"
        else:
            return truncated + "\n[...texto truncado...]"
    
    def summarize_single_paper(
        self, 
        documents: List[Document], 
        focus: str = "metodologia"
    ) -> Dict[str, Any]:
        
        if not documents:
            return {
                "summary": "",
                "metadata": {},
                "focus": focus,
                "success": False,
                "error": "Nenhum documento fornecido"
            }
        
        # Combina todos os chunks do paper
        full_text = "\n\n".join([doc.page_content for doc in documents])
        full_text = self._truncate_text(full_text, max_chars=10000)
        
        # Pega metadados do primeiro chunk
        metadata = documents[0].metadata if documents else {}
        source_file = metadata.get("source_file", "Documento desconhecido")
        
        # Define prompts especializados por foco
        focus_prompts = {
            "metodologia": """
Analise APENAS a metodologia deste paper acadêmico:

{text}

Extraia e estruture:

**1. Tipo de Estudo**
- Classificação (experimental, observacional, revisão sistemática, etc)

**2. Amostra/População**
- Tamanho amostral
- Critérios de inclusão/exclusão
- Características principais

**3. Métodos e Técnicas**
- Procedimentos utilizados
- Instrumentos/ferramentas
- Protocolos seguidos

**4. Análise de Dados**
- Testes estatísticos aplicados
- Software utilizado
- Nível de significância adotado

Seja técnico mas conciso (máx 200 palavras).
""",
            "resultados": """
Analise APENAS os resultados deste paper acadêmico:

{text}

Extraia:

**1. Principais Achados**
- Top 3 resultados mais relevantes
- Dados quantitativos (percentuais, médias, p-valores)

**2. Tabelas/Figuras Chave**
- Principais dados apresentados

**3. Significância dos Resultados**
- O que os números indicam
- Relação com hipóteses do estudo

**4. Resultados Secundários**
- Achados adicionais relevantes

Seja preciso com números. Máx 200 palavras.
""",
            "limitacoes": """
Analise as limitações e gaps deste paper acadêmico:

{text}

Identifique:

**1. Limitações Metodológicas**
- Problemas de design/execução
- Vieses potenciais

**2. Limitações Amostrais**
- Tamanho amostral insuficiente
- Problemas de representatividade

**3. Gaps de Pesquisa**
- O que o estudo NÃO conseguiu responder
- Sugestões para pesquisas futuras

**4. Implicações Práticas**
- Como as limitações afetam a aplicabilidade

Seja crítico mas construtivo. Máx 150 palavras.
""",
            "completo": """
Faça um resumo executivo completo deste paper acadêmico:

{text}

Estruture em:

**1. Contextualização**
- Problema de pesquisa
- Relevância do tema

**2. Objetivo**
- O que o estudo se propôs a fazer

**3. Metodologia** (resumo)
- Tipo de estudo
- Amostra
- Principais técnicas

**4. Resultados** (principais achados)
- Top 3 descobertas
- Dados quantitativos chave

**5. Conclusões**
- Implicações principais
- Contribuição do estudo

Seja completo mas conciso. Máx 250 palavras.
"""
        }
        
        prompt_template = focus_prompts.get(focus, focus_prompts["completo"])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um revisor acadêmico experiente especializado em análise crítica de literatura científica.
Sua análise deve ser:
- Técnica e precisa
- Baseada estritamente no texto fornecido
- Estruturada e organizada
- Sem especulações além do texto"""),
            ("human", prompt_template)
        ])
        
        chain = prompt | self.llm
        
        try:
            print(f"      ⏳ Analisando {focus}: {source_file[:50]}...")
            
            response = chain.invoke({"text": full_text})
            summary = response.content if hasattr(response, 'content') else str(response)
            
            print(f"      ✅ Análise concluída ({len(summary)} chars)")
            
            return {
                "summary": summary,
                "metadata": metadata,
                "focus": focus,
                "success": True,
                "error": None,
                "word_count": len(summary.split())
            }
        
        except Exception as e:
            print(f"      ❌ Erro na análise: {str(e)}")
            return {
                "summary": "",
                "metadata": metadata,
                "focus": focus,
                "success": False,
                "error": str(e),
                "word_count": 0
            }
    
    def compare_papers(
        self,
        summaries: List[Dict[str, Any]],
        comparison_focus: str = "metodologia"
    ) -> str:
        """
        Compara múltiplos papers baseado em seus resumos.
        
        Fase REDUCE do Map-Reduce.
        
        Args:
            summaries: Lista de resumos gerados por summarize_single_paper
            comparison_focus: Aspecto a comparar
            
        Returns:
            Texto de síntese comparativa em Markdown
        """
        if not summaries:
            return "⚠️ Nenhum resumo disponível para comparação."
        
        # Filtra apenas resumos bem-sucedidos
        valid_summaries = [s for s in summaries if s["success"]]
        
        if not valid_summaries:
            return "⚠️ Nenhum resumo válido para comparação."
        
        if len(valid_summaries) < 2:
            return "⚠️ Necessário pelo menos 2 papers para comparação."
        
        # Constrói texto com todos os resumos
        summaries_text = ""
        for i, summary in enumerate(valid_summaries, 1):
            meta = summary["metadata"]
            author = meta.get("author", "Autor desconhecido")
            year = meta.get("year", "Ano desconhecido")
            source = meta.get("source_file", "Documento")
            
            summaries_text += f"\n\n### Paper {i}: {author} ({year})\n**Fonte:** {source}\n\n{summary['summary']}"
        
        # Mapeia focos para títulos bonitos
        focus_titles = {
            "metodologia": "Metodologias",
            "resultados": "Resultados",
            "limitacoes": "Limitações",
            "completo": "Aspectos Gerais"
        }
        
        focus_title = focus_titles.get(comparison_focus, comparison_focus.title())
        
        # Prompt de comparação SIMPLIFICADO
        comparison_prompt = f"""
            Compare estes {len(valid_summaries)} papers sobre {focus_title}:

            {summaries_text}

            Destaque: semelhanças, diferenças e lacunas de pesquisa.
            Máximo 250 palavras. Formato Markdown.
            """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é um revisor sênior especializado em síntese de literatura científica."),
            ("human", comparison_prompt)
        ])
        
        chain = prompt | self.llm
        
        try:
            print(f"   ⏳ Gerando síntese comparativa...")
            response = chain.invoke({})
            synthesis = response.content if hasattr(response, 'content') else str(response)
            print(f"   ✅ Síntese gerada ({len(synthesis)} chars)")
            return synthesis
        
        except Exception as e:
            print(f"   ❌ Erro ao gerar síntese: {str(e)}")
            return f"❌ **Erro ao gerar comparação:** {str(e)}"
    
    def generate_literature_review(
        self,
        papers_documents: Dict[str, List[Document]],
        focus: str = "completo",
        include_individual: bool = True
    ) -> Dict[str, Any]:
        """
        Gera uma revisão de literatura completa (Map-Reduce completo).
        
        Args:
            papers_documents: Dicionário {nome_paper: [chunks]}
            focus: Aspecto a focar na revisão
            include_individual: Se True, inclui resumos individuais no resultado
            
        Returns:
            Dicionário com revisão completa
        """
        start_time = datetime.now()
        
        print(f"\n📚 Iniciando revisão de literatura")
        print(f"   📄 Papers a analisar: {len(papers_documents)}")
        print(f"   🎯 Foco: {focus}")
        print(f"   ⏰ Início: {start_time.strftime('%H:%M:%S')}")
        
        # FASE MAP: Resume cada paper individualmente
        print(f"\n   === FASE MAP: Análise Individual ===")
        summaries = []
        
        for i, (paper_name, documents) in enumerate(papers_documents.items(), 1):
            print(f"\n   [{i}/{len(papers_documents)}] {paper_name}")
            summary = self.summarize_single_paper(documents, focus=focus)
            summaries.append(summary)
        
        successful = sum(1 for s in summaries if s["success"])
        print(f"\n   ✅ MAP concluído: {successful}/{len(summaries)} papers analisados com sucesso")
        
        # FASE REDUCE: Compara todos os resumos
        print(f"\n   === FASE REDUCE: Síntese Comparativa ===")
        comparison = self.compare_papers(summaries, comparison_focus=focus)
        
        # Calcula métricas
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        total_words = sum(s.get("word_count", 0) for s in summaries if s["success"])
        
        print(f"\n   ✅ REDUCE concluído")
        print(f"   ⏱️ Tempo total: {duration:.1f}s")
        print(f"   📊 Palavras geradas: {total_words}")
        
        result = {
            "comparative_synthesis": comparison,
            "total_papers": len(papers_documents),
            "successful_analyses": successful,
            "focus": focus,
            "duration_seconds": duration,
            "generated_at": datetime.now().isoformat(),
            "total_words": total_words
        }
        
        if include_individual:
            result["individual_summaries"] = summaries
        
        return result
    
    def export_to_markdown(self, review_result: Dict[str, Any]) -> str:
        """
        Exporta revisão de literatura para Markdown formatado.
        
        Args:
            review_result: Resultado de generate_literature_review()
            
        Returns:
            String em Markdown
        """
        focus_names = {
            "metodologia": "Metodologias",
            "resultados": "Resultados",
            "limitacoes": "Limitações",
            "completo": "Revisão Completa"
        }
        
        focus_title = focus_names.get(review_result["focus"], review_result["focus"].title())
        
        markdown = f"""# 📚 Revisão de Literatura: {focus_title}

**Gerado em:** {datetime.fromisoformat(review_result["generated_at"]).strftime("%d/%m/%Y às %H:%M")}  
**Papers analisados:** {review_result["successful_analyses"]}/{review_result["total_papers"]}  
**Tempo de processamento:** {review_result["duration_seconds"]:.1f}s

---

{review_result["comparative_synthesis"]}

---

"""
        
        # Adiciona resumos individuais se disponíveis
        if "individual_summaries" in review_result:
            markdown += "\n\n# 📄 Resumos Individuais\n\n"
            
            for i, summary in enumerate(review_result["individual_summaries"], 1):
                if summary["success"]:
                    meta = summary["metadata"]
                    author = meta.get("author", "Autor desconhecido")
                    year = meta.get("year", "?")
                    source = meta.get("source_file", "Documento")
                    
                    markdown += f"""
## Paper {i}: {author} ({year})

**Fonte:** `{source}`

{summary["summary"]}

---

"""
        
        markdown += f"""
---
*Revisão gerada automaticamente pelo Assistente Acadêmico com IA*  
*Modelo: Llama 3.3 70B via Groq | Embedding: sentence-transformers*
"""
        
        return markdown


# Função auxiliar para uso rápido
def quick_literature_review(
    llm: ChatGroq, 
    papers_docs: Dict[str, List[Document]],
    focus: str = "completo"
) -> str:
    """
    Revisão de literatura rápida (retorna apenas síntese em Markdown).
    
    Usage:
        review_md = quick_literature_review(llm, {
            "paper1.pdf": chunks1,
            "paper2.pdf": chunks2
        }, focus="metodologia")
    """
    synthesizer = PaperSynthesizer(llm)
    result = synthesizer.generate_literature_review(papers_docs, focus=focus)
    return synthesizer.export_to_markdown(result)
"""
Assistente Acadêmico com RAG - Versão MVP
Fase 1: Estrutura básica funcional
"""
import streamlit as st
import os
from config import UI_CONFIG, LLM_CONFIG, EMBEDDING_CONFIG, EMBEDDING_OPTIONS, DEFAULT_EMBEDDING, CHROMA_DIR
import datetime


current_year = datetime.date.today().year

# Configuração da página
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"]
)

# Cabeçalho
st.title("🎓 Assistente Acadêmico com IA")
st.caption("Análise inteligente de papers científicos usando RAG")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Input da API Key
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Obtenha sua chave em: https://console.groq.com/"
    )
    
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.success("✅ API Key configurada")
    
    st.divider()

    st.subheader("🧠 Modelo de Embedding")

    from config import EMBEDDING_OPTIONS, DEFAULT_EMBEDDING

    selected_embedding = st.selectbox(
        "Escolha o modelo de embedding",
        options=list(EMBEDDING_OPTIONS.keys()),
        index=list(EMBEDDING_OPTIONS.keys()).index(DEFAULT_EMBEDDING),
        help="Diferentes modelos têm trade-offs entre velocidade e qualidade"
        )
    
    embedding_info = EMBEDDING_OPTIONS[selected_embedding]
    
    with st.expander("ℹ️ Detalhes do Modelo", expanded=False):
        st.write(f"**Descrição:** {embedding_info['description']}")
        st.write(f"**Dimensões:** {embedding_info['dimensions']}")
        st.write(f"**Velocidade:** {embedding_info['speed']}")
        st.write(f"**Qualidade:** {embedding_info['quality']}")
    
    st.session_state.selected_embedding_model = embedding_info["model_name"]
    st.session_state.selected_embedding_name = selected_embedding
    
    st.divider()
    
    st.subheader("📚 Sobre o Projeto")
    st.info(
        """
        Este assistente utiliza **RAG (Retrieval-Augmented Generation)** 
        para analisar papers científicos e responder perguntas sobre:
        
        - Metodologias utilizadas
        - Resultados principais
        - Comparação entre estudos
        - Síntese de literatura
        """
    )
    
    st.divider()
    
    st.markdown("**Status do Sistema:**")
    st.write(f"📦 Modelo: `{LLM_CONFIG['model']}`")
    st.write(f"🌡️ Temperatura: `{LLM_CONFIG['temperature']}`")

# Verificação de API Key
if not api_key:
    st.warning("⚠️ Configure sua Groq API Key na barra lateral para começar")
    st.stop()

# Área principal (por enquanto apenas placeholder)
st.markdown("---")
st.subheader("📄 Upload de Papers")

uploaded_files = st.file_uploader(
    "Faça upload de um ou mais papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Você pode enviar múltiplos PDFs para análise comparativa"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} arquivo(s) carregado(s)")
    
    st.markdown("### 📋 Metadados dos Papers")
    st.info("💡 Preencha os metadados para melhorar as buscas por autor/ano. Deixe em branco se não souber.")

    # incializa dicionário de metadados se não existr
    if "manual_metadata" not in st.session_state:
        st.session_state.manual_metadata = {}

    #cria formulário para cada arquivo
    metadata_forms = []
    for i, uploaded_file in enumerate(uploaded_files):
        with st.expander(f"📄 {uploaded_file.name}", expanded=(i==0)):
            col1, col2, col3 = st.columns(3)

            with col1:
                author = st.text_input(
                    "Primeiro Autor (sobrenome)",
                    key=f"author_{i}",
                    placeholder="Ex: Silva",
                    help="Sobrenome do primeiro autor"
                )

            with col2:
                metadata = st.session_state.manual_metadata.get(uploaded_file.name, {})
                year = st.number_input(
                    "Ano de Publicação",
                    min_value=0,              # Permite papers históricos
                    max_value=current_year,   # Bloqueia anos no futuro
                    value=metadata.get("year", current_year), # Usa o ano extraído pelo Llama ou o atual
                    key=f"year_{i}",
                    help="Ano de publicação do paper"
                )
            with col3:
                title = st.text_input(
                    "Titulo (opc)",
                    key=f"title_{i}",
                    placeholder="Ex: Machie=ne Learning...",
                    help="Título do paper (opcional)"
                )

            #salva no session_state
            st.session_state.manual_metadata[uploaded_file.name] = {
                "author": author if author else None,
                "year": year,
                "title": title if title else None
            }
    st.divider()


    if st.button("📊 Processar Documentos", type="primary"):

        # Importa o processador
        from src.document_processor import DocumentProcessor
        from src.utils import get_document_stats

        #Inicializa o processador
        processor = DocumentProcessor()
        #Processa cada PDF
        all_results = []

        with st.spinner("Processando PDFs..."):
            for uploaded_file in uploaded_files:
                result = processor.process_pdf(uploaded_file, uploaded_file.name)
                all_results.append(result)

            manual_meta = st.session_state.manual_metadata.get(uploaded_file.name, {})

            for key in ["author", "year", "title"]:
                if value := manual_meta.get(key):
                    result["metadata"][key] = value
            
            # Atualiza chunks com metadados corrigidos
            if result["success"]:
                updates = {k: v for k, v in manual_meta.items() if v and k in ["author", "year", "title"]}
                for chunk in result["documents"]:
                    chunk.metadata |= updates
        
        # Exibe resultados
        st.markdown("### 📈 Resultados do Processamento")

        for i, result in enumerate(all_results, 1):
            if result["success"]:
                st.success(f"✅ **{result['metadata']['source_file']}**")
        
         #Mostra metadados extraídos
                meta = result["metadata"]
                display_map = {"author": "👤", "year": "📅", "title": "📖"}

                parts = [f"{display_map[k]} {str(meta[k])[:50]}" for k in display_map if meta.get(k)]

                if parts:
                    st.write("**Metadados:**")
                    st.caption(" | ".join(parts))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Páginas", result["stats"]["total_pages"])
                with col2:
                    st.metric("Chunks", result["stats"]["total_chunks"])
                with col3:
                    st.metric("média/Chunk", f"{result['stats']['avg_chunk_size']:.0f} chars")

                #Mostra Preview dos primeiros chunks
                with st.expander("🔍 Preview dos Chunks"):
                    from src.utils import format_document_for_display

                    for j, doc in enumerate(result["documents"][:3], 1):
                        st.markdown(f"**Chunk {j}:**")
                        st.text(format_document_for_display(doc, max_length=300))
                        st.divider()

            else:
                st.error(f"❌ **{uploaded_files[i-1].name}**: {result['error']}")

        st.session_state.processed_docs = all_results
        st.session_state.processing_done = True
        


    # Mostrar lista de arquivos
    with st.expander("📋 Arquivos Carregados"):
        for i, file in enumerate(uploaded_files, 1):
            st.write(f"{i}. {file.name} ({file.size / 1024:.1f} KB)")
else:
    st.info("👆 Comece fazendo upload de papers científicos")

if st.session_state.get("processing_done"):
            st.markdown("---")
            st.subheader("🔧 Configurar Sistema RAG")

            # Mostra qual embedding está selecionado
            if st.session_state.get("selected_embedding_name"):
                st.info(f"🧠 Modelo selecionado: **{st.session_state.selected_embedding_name}**")
    

            if st.button("⚙️ Criar Banco Vetorial", type="primary"):
                from src.rag_engine import RAGEngine

                #coleta todos os chunks proessados
                all_chunks = []
                for result in st.session_state.processed_docs:
                    if result["success"]:
                        all_chunks.extend(result["documents"])

                if not all_chunks:
                    st.error("Nenhum documento processado com sucesso")
                    st.stop()

                embedding_model = st.session_state.get(
                    "selected_embedding_model",
                    EMBEDDING_CONFIG["model_name"]
                )
                 #Cria nome de collection único baseado no embedding
                # Isso permite ter múltiplas versões com embeddings diferentes
                embedding_collection_map = {
                    "MiniLM (Rápido)": "papers_minilm",
                    "Nomic Embed (Balanceado)": "papers_nomic",
                    "BGE-M3 (Premium)": "papers_bge_m3"
                }

                collection_name = embedding_collection_map.get(
                    st.session_state.selected_embedding_name,
                    "papers_default"
)

                with st.spinner(f"Criando banco vetorial com {len(all_chunks)} chunks..."):
                    try:
                        # Cria o motor RAG
                        rag_engine = RAGEngine(
                            embedding_model=embedding_model,
                            collection_name=collection_name
                    )
                        
                        # Cria vectorstore
                        rag_engine.create_vectorstore(all_chunks)
                        
                        # Cria retriever
                        rag_engine.create_retriever()
                        
                        # Salva no session_state
                        st.session_state.rag_engine = rag_engine
                        st.session_state.rag_ready = True
                        st.session_state.current_embedding = st.session_state.selected_embedding_name
                        st.success("✅ Sistema RAG criado com sucesso!")
                        
                        # Mostra estatísticas
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total de Chunks", len(all_chunks))
                        with col2:
                            st.metric("Embedding", st.session_state.selected_embedding_name.split()[0])
                        with col3:
                            st.metric("Dimensões", EMBEDDING_OPTIONS[st.session_state.selected_embedding_name]["dimensions"])
                        with col4:
                            st.metric("Collection", collection_name[:15] + "...")
                    except Exception as e:
                        st.error(f"Erro ao criar sistema RAG: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                        st.stop()
                
# Área de perguntas (CORRIGIDA)
st.markdown("---")
st.subheader("💬 Faça sua Pergunta")

# Inicializa variáveis de filtro
use_author_filter = False
use_year_filter = False
selected_author = None
selected_year = None

if not st.session_state.get("rag_ready"):
    st.info("👆 Processe os documentos e crie o banco vetorial primeiro")
    pergunta = st.text_area(
        "Digite sua pergunta sobre os papers",
        height=100,
        disabled=True,
        placeholder="Configure o sistema RAG primeiro..."
    )
else:
    # FILTROS DE BUSCA (só aparece se RAG está pronto)
    st.markdown("#### 🔍 Filtros de Busca (Opcional)")
    
    # Coletar autores e anos disponíveis
    available_authors = set()
    available_years = set()
    
    if st.session_state.get("processed_docs"):
        for result in st.session_state.processed_docs:
            if result["success"]:
                meta = result["metadata"]
                if meta.get("author"):
                    available_authors.add(meta["author"])
                if meta.get("year"):
                    available_years.add(meta["year"])
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        use_author_filter = st.checkbox("Filtrar por Autor")
        if use_author_filter and available_authors:
            selected_author = st.selectbox(
                "Selecione o autor",
                options=["Todos"] + sorted(list(available_authors)),
                key="filter_author"
            )
            if selected_author == "Todos":
                selected_author = None
    
    with filter_col2:
        use_year_filter = st.checkbox("Filtrar por Ano")
        if use_year_filter and available_years:
            selected_year = st.selectbox(
                "Selecione o ano",
                options=["Todos"] + sorted(list(available_years), reverse=True),
                key="filter_year"
            )
            if selected_year == "Todos":
                selected_year = None
    
    with filter_col3:
        st.write("")  # Espaçamento
        st.caption("💡 Use filtros para comparar estudos específicos")
    
    st.divider()
    
    # TEXT AREA DE PERGUNTA
    pergunta = st.text_area(
        "Digite sua pergunta sobre os papers",
        height=100,
        placeholder="Ex: Quais metodologias foram utilizadas?"
    )

# BOTÃO DE ANALISAR (fora do if/else para sempre estar disponível)
btn_col1, btn_col2 = st.columns([1, 5])
with btn_col1:
    btn_perguntar = st.button(
        "🔍 Analisar", 
        type="primary", 
        disabled=not st.session_state.get("rag_ready")
    )

# PROCESSAMENTO DA PERGUNTA
if btn_perguntar and pergunta:
    with st.spinner("🤔 Analisando papers e gerando resposta..."):
        try:
            # Determina se usa filtros
            author_filter = selected_author if (use_author_filter and selected_author) else None
            year_filter = int(selected_year) if (use_year_filter and selected_year) else None
            
            # Faz query com ou sem filtros
            if author_filter or year_filter:
                result = st.session_state.rag_engine.query_with_filters(
                    question=pergunta,
                    author=author_filter,
                    year=year_filter,
                    return_sources=True
                )
                
                # Mostra filtros aplicados
                filters_info = []
                if author_filter:
                    filters_info.append(f"👤 Autor: **{author_filter}**")
                if year_filter:
                    filters_info.append(f"📅 Ano: **{year_filter}**")
                st.info("🔍 Filtros aplicados: " + " | ".join(filters_info))
            else:
                result = st.session_state.rag_engine.query(
                    question=pergunta,
                    return_sources=True
                )

            with st.expander("🐛 DEBUG - Chunks Recuperados (clique para ver)"):
                for i, doc in enumerate(result["sources"], 1):
                    st.markdown(f"**Chunk {i} (score de similaridade):**")
                    st.markdown(f"- **Arquivo:** {doc.metadata.get('source_file', 'N/A')}")
                    st.markdown(f"- **Página:** {doc.metadata.get('page', '?')}")
                    st.markdown(f"- **Autor:** {doc.metadata.get('author', 'N/A')}")
                    st.text(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)
                    st.divider()

            # Exibe resposta
            st.markdown("### 📝 Resposta")
            st.write(result["answer"])
            
            # Exibe fontes usadas
            st.markdown("---")
            st.markdown("### 📚 Fontes Consultadas")
            
            for i, doc in enumerate(result["sources"], 1):
                with st.expander(f"📄 Fonte {i}: {doc.metadata.get('source_file', 'N/A')} - Página {doc.metadata.get('page', '?')}"):
                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            
            # Exibe metadados
            with st.expander("ℹ️ Informações da Consulta"):
                st.json(result["metadata"])
        
        except Exception as e:
            st.error(f"Erro ao processar pergunta: {str(e)}")

# ==================== SÍNTESE DE LITERATURA (FEATURE PRINCIPAL) ====================
if st.session_state.get("rag_ready") and st.session_state.get("processed_docs"):
    st.markdown("---")
    st.markdown("## 📚 Síntese de Literatura Automatizada")
    
    st.info("""
    **🎯 Feature Diferenciada:** Análise comparativa automática de múltiplos papers acadêmicos.
    
    **Como funciona:**
    1. **MAP:** Cada paper é analisado individualmente focando no aspecto escolhido
    2. **REDUCE:** Os resumos são sintetizados em uma comparação estruturada
    3. **EXPORT:** Resultado disponível em Markdown/TXT para uso em trabalhos acadêmicos
    """)
    
    # Verifica quantos papers foram processados
    total_papers = len([r for r in st.session_state.processed_docs if r["success"]])
    
    if total_papers < 2:
        st.warning(f"⚠️ Você tem apenas {total_papers} paper(s). Carregue pelo menos 2 para comparação.")
    else:
        st.success(f"✅ {total_papers} papers prontos para síntese comparativa")
        
        # Configurações da síntese
        col1, col2 = st.columns([2, 1])
        
        with col1:
            synthesis_focus = st.selectbox(
                "🎯 Foco da Análise",
                options=["completo", "metodologia", "resultados", "limitacoes"],
                help="Escolha o aspecto que deseja comparar entre os papers"
            )
        
        with col2:
            include_individual = st.checkbox(
                "Incluir resumos individuais",
                value=True,
                help="Além da síntese comparativa, incluir resumo de cada paper"
            )
        
        # Descrições dos focos
        focus_descriptions = {
            "completo": "📖 **Revisão Completa:** Objetivo, metodologia, resultados e conclusões de cada paper",
            "metodologia": "🔬 **Metodologias:** Foca em métodos, técnicas, amostras e análises estatísticas",
            "resultados": "📊 **Resultados:** Foca em achados principais, dados quantitativos e significância",
            "limitacoes": "⚠️ **Limitações:** Foca em problemas metodológicos e gaps de pesquisa"
        }
        
        st.markdown(focus_descriptions[synthesis_focus])
        
        st.divider()
        
        # Botão principal
        if st.button("🚀 Gerar Revisão de Literatura", type="primary", use_container_width=True):
            from src.synthesis import PaperSynthesizer
            
            # Agrupa chunks por paper
            papers_documents = {}
            for result in st.session_state.processed_docs:
                if result["success"]:
                    source_file = result["metadata"]["source_file"]
                    papers_documents[source_file] = result["documents"]
            
            # Estimativa de tempo
            estimated_time = len(papers_documents) * 15  # ~15s por paper
            
            with st.spinner(f"⏳ Processando {len(papers_documents)} papers... (tempo estimado: ~{estimated_time}s)"):
                try:
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Callback para atualizar progresso (simulado)
                    import time
                    
                    status_text.text("📖 Fase MAP: Analisando papers individuais...")
                    progress_bar.progress(0.2)
                    
                    # Cria sintetizador
                    synthesizer = PaperSynthesizer(st.session_state.rag_engine.llm)
                    
                    # Gera revisão
                    review = synthesizer.generate_literature_review(
                        papers_documents,
                        focus=synthesis_focus,
                        include_individual=include_individual
                    )
                    
                    progress_bar.progress(0.8)
                    status_text.text("🔄 Fase REDUCE: Gerando síntese comparativa...")
                    
                    # Exporta para Markdown
                    markdown_output = synthesizer.export_to_markdown(review)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Revisão de literatura concluída!")
                    
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Salva no session_state para não perder
                    st.session_state.last_review = {
                        "result": review,
                        "markdown": markdown_output,
                        "timestamp": datetime.datetime.now()
                    }
                    
                    # Métricas da análise
                    st.markdown("### 📊 Métricas da Análise")
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric("Papers Analisados", f"{review['successful_analyses']}/{review['total_papers']}")
                    with metric_col2:
                        st.metric("Tempo de Processamento", f"{review['duration_seconds']:.1f}s")
                    with metric_col3:
                        st.metric("Palavras Geradas", review['total_words'])
                    with metric_col4:
                        focus_emoji = {"completo": "📖", "metodologia": "🔬", "resultados": "📊", "limitacoes": "⚠️"}
                        st.metric("Foco", f"{focus_emoji.get(synthesis_focus, '📄')} {synthesis_focus.title()}")
                    
                    st.divider()
                    
                    # Exibe síntese comparativa
                    st.markdown("### 📊 Síntese Comparativa")
                    st.markdown(review["comparative_synthesis"])
                    
                    # Resumos individuais (se solicitado)
                    if include_individual and "individual_summaries" in review:
                        st.markdown("---")
                        st.markdown("### 📄 Resumos Individuais")
                        
                        for i, summary in enumerate(review["individual_summaries"], 1):
                            if summary["success"]:
                                meta = summary["metadata"]
                                author = meta.get("author", "Autor desconhecido")
                                year = meta.get("year", "?")
                                source = meta.get("source_file", "Documento")
                                
                                with st.expander(f"📑 Paper {i}: {author} ({year}) - {source[:40]}..."):
                                    st.markdown(summary["summary"])
                                    st.caption(f"💬 {summary['word_count']} palavras")
                    
                    # Botões de export
                    st.markdown("---")
                    st.markdown("### ⬇️ Exportar Revisão")
                    
                    export_col1, export_col2, export_col3 = st.columns(3)
                    
                    with export_col1:
                        st.download_button(
                            "📝 Download Markdown",
                            data=markdown_output,
                            file_name=f"revisao_literatura_{synthesis_focus}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                    
                    with export_col2:
                        st.download_button(
                            "📄 Download TXT",
                            data=markdown_output,
                            file_name=f"revisao_literatura_{synthesis_focus}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with export_col3:
                        # Copia para clipboard (via botão)
                        if st.button("📋 Copiar Texto", use_container_width=True):
                            st.toast("✅ Texto copiado! Use Ctrl+V para colar")
                            st.code(markdown_output[:500] + "\n...\n[Use o botão de download para texto completo]")
                
                except Exception as e:
                    st.error(f"❌ Erro ao gerar revisão: {str(e)}")
                    import traceback
                    with st.expander("🐛 Detalhes do Erro (para debug)"):
                        st.code(traceback.format_exc())

# Mostra última revisão gerada (se houver)
if st.session_state.get("last_review"):
    with st.expander("🕒 Última Revisão Gerada", expanded=False):
        last = st.session_state.last_review
        st.caption(f"Gerada em: {last['timestamp'].strftime('%d/%m/%Y às %H:%M:%S')}")
        
        st.download_button(
            "⬇️ Re-download da Última Revisão",
            data=last["markdown"],
            file_name=f"revisao_ultima.md",
            mime="text/markdown"
        )

# Footer
st.markdown("---")
st.caption("Desenvolvido para portfólio de João Otávio Mochiuti | Powered by LangChain + Llama 3.3 70B via Groq")
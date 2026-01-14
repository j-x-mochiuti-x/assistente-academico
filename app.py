"""
Assistente Acadêmico com RAG - Versão MVP
Fase 1: Estrutura básica funcional
"""
import streamlit as st
import os
from config import UI_CONFIG, LLM_CONFIG

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
        
        # Exibe resultados
        st.markdown("### 📈 Resultados do Processamento")

        for i, result in enumerate(all_results, 1):
            if result["success"]:
                st.success(f"✅ **{result['metadata']['source_file']}**")

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

                with st.spinner(f"Criando banco vetorial com {len(all_chunks)} chunks..."):
                    try:
                        # Cria o motor RAG
                        rag_engine = RAGEngine()
                        
                        # Cria vectorstore
                        rag_engine.create_vectorstore(all_chunks)
                        
                        # Cria retriever
                        rag_engine.create_retriever()
                        
                        # Salva no session_state
                        st.session_state.rag_engine = rag_engine
                        st.session_state.rag_ready = True
                        
                        st.success("✅ Sistema RAG criado com sucesso!")
                        
                        # Mostra estatísticas
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total de Chunks", len(all_chunks))
                        with col2:
                            st.metric("Modelo Embedding", rag_engine.embedding_model_name.split('/')[-1])
                        with col3:
                            st.metric("Modelo LLM", rag_engine.llm_model_name.split('/')[-1])
                    
                    except Exception as e:
                        st.error(f"Erro ao criar sistema RAG: {str(e)}")
                        st.stop()
                
# Área de perguntas (atualizada)
st.markdown("---")
st.subheader("💬 Faça sua Pergunta")

if not st.session_state.get("rag_ready"):
    st.info("👆 Processe os documentos e crie o banco vetorial primeiro")
    pergunta = st.text_area(
        "Digite sua pergunta sobre os papers",
        height=100,
        disabled=True,
        placeholder="Configure o sistema RAG primeiro..."
    )
else:
    pergunta = st.text_area(
        "Digite sua pergunta sobre os papers",
        height=100,
        placeholder="Ex: Quais metodologias foram utilizadas nos estudos?"
    )

col1, col2 = st.columns([1, 5])
with col1:
    btn_perguntar = st.button(
        "🔍 Analisar", 
        type="primary", 
        disabled=not st.session_state.get("rag_ready")
    )

if btn_perguntar and pergunta:
    with st.spinner("🤔 Analisando papers e gerando resposta..."):
        try:
            # Faz a query no sistema RAG
            result = st.session_state.rag_engine.query(
                question=pergunta,
                return_sources=True
            )
            
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

# Footer
st.markdown("---")
st.caption("Desenvolvido para portfólio de João Otávio Mochiuti | Powered by LangChain + Groq")
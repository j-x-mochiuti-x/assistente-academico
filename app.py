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
    
    # Mostrar lista de arquivos
    with st.expander("📋 Arquivos Carregados"):
        for i, file in enumerate(uploaded_files, 1):
            st.write(f"{i}. {file.name} ({file.size / 1024:.1f} KB)")
else:
    st.info("👆 Comece fazendo upload de papers científicos")

st.markdown("---")
st.subheader("💬 Faça sua Pergunta")

pergunta = st.text_area(
    "Digite sua pergunta sobre os papers",
    height=100,
    placeholder="Ex: Quais metodologias foram utilizadas nos estudos sobre machine learning?"
)

col1, col2 = st.columns([1, 5])
with col1:
    btn_perguntar = st.button("🔍 Analisar", type="primary", disabled=not uploaded_files)

if btn_perguntar and pergunta:
    st.info("🚧 Funcionalidade em desenvolvimento - Fase 2")

# Footer
st.markdown("---")
st.caption("Desenvolvido para portfólio de Ciência de Dados | Powered by LangChain + Groq")
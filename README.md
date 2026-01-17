# 🎓 Assistente Acadêmico com RAG

Sistema de análise inteligente de papers científicos utilizando Retrieval-Augmented Generation (RAG).

## 👨‍💻 Autor

João Otávio Mochiuti - Cientista de Dados em formação

## 📋 Status do Projeto

- [x] Fase 1: Setup e estrutura base
- [x] Fase 2: Processamento de documentos
- [x] Fase 3: Pipeline RAG funcional
- [ ] Fase 4: Funcionalidades avançadas
- [ ] Fase 5: Deploy e documentação final

## 🗺️ Roadmap de Funcionalidades

### ✅ Fase 3 (Concluída)
- [x] Upload e processamento de PDFs
- [x] Extração e indexação vetorial (ChromaDB)
- [x] Sistema RAG completo
- [x] Filtros por autor/ano
- [x] Múltiplos modelos de embedding
- [x] Preview de síntese de literatura

### 🚧 Fase 4 (Em Desenvolvimento)
- [ ] Síntese de literatura completa (Map-Reduce)
- [ ] Comparação automática de metodologias
- [ ] Export de revisões em PDF/Word
- [ ] Hybrid Search (semântico + keyword)
- [ ] Reranking de resultados
- [ ] Visualizações interativas

### 📋 Fase 5 (Planejada)
- [ ] Deploy em produção
- [ ] Testes automatizados completos
- [ ] Documentação técnica detalhada
- [ ] Otimizações de performance

## 🚀 Como Executar (Fase 1)
```bash
# 1. Clone o repositório
git clone <seu-repo>
cd assistente-academico

# 2. Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Instale dependências
pip install -r requirements.txt

# 4. Execute o app
streamlit run app.py
```

## 🔑 Configuração

Obtenha sua API Key em: https://console.groq.com/

## 📚 Tecnologias

- **Streamlit**: Interface web
- **LangChain**: Framework RAG
- **ChromaDB**: Banco vetorial
- **Groq**: LLM inference
- **HuggingFace**: Embeddings

## 🧠 Modelos de Embedding Suportados

- **MiniLM-L6-v2**: Rápido e eficiente (recomendado para testes)
- **Nomic Embed v1.5**: Melhor qualidade (recomendado para produção) ⭐
- **BGE-M3**: Máxima qualidade (requer GPU ou *PACIÊNCIA!!*)

### Benchmark Interno
Pergunta: "Qual é o objetivo do trabalho?"
- MiniLM: ✅ Recuperou corretamente (0.5s indexação)
- Nomic: ✅ Recuperou + melhor estruturação (2.1s indexação)



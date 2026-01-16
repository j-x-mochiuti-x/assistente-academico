# 🎓 Assistente Acadêmico com RAG

Sistema de análise inteligente de papers científicos utilizando Retrieval-Augmented Generation (RAG).

## 👨‍💻 Autor

João Otávio Mochiuti - Cientista de Dados em formação

## 📋 Status do Projeto

- [x] Fase 1: Setup e estrutura base
- [ ] Fase 2: Processamento de documentos
- [ ] Fase 3: Pipeline RAG funcional
- [ ] Fase 4: Funcionalidades avançadas
- [ ] Fase 5: Deploy e documentação final

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

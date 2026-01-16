"""
Motor RAG (Retrieval-Augmented Generation).
Responsável por:
1. Criar banco vetorial (embeddings)
2. Buscar documentos relevantes
3. Gerar respostas usando LLM + contexto
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from config import EMBEDDING_CONFIG, LLM_CONFIG, RETRIEVAL_CONFIG, CHROMA_DIR

class RAGEngine:
    """
    Motor RAG completo para análise de papers acadêmicos.
    """
    
    def __init__(
        self, 
        embedding_model: str = None,
        llm_model: str = None,
        persist_directory: str = None,
        collection_name: str = "academic_papers"
    ):
        # Configurações
        self.embedding_model_name = embedding_model or EMBEDDING_CONFIG["model_name"]
        self.llm_model_name = llm_model or LLM_CONFIG["model"]
        self.persist_dir = persist_directory or str(CHROMA_DIR)
        self.collection_name = collection_name
        
        # Inicializa componentes (lazy loading - só cria quando necessário)
        self._embeddings = None
        self._llm = None
        self._vectorstore = None
        self._retriever = None
        
        print(f"✅ RAGEngine inicializado")
        print(f"   📦 Embedding: {self.embedding_model_name}")
        print(f"   🤖 LLM: {self.llm_model_name}")
    
    @property
    def embeddings(self):
        """
        Lazy loading: só carrega embeddings quando necessário.
        Embeddings são modelos pesados (80MB-2GB), economiza memória.
        """
        if self._embeddings is None:
            print(f"⏳ Carregando modelo de embeddings: {self.embedding_model_name}...")
            
            # Desativa paralelismo do tokenizador (evita warnings)
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},  # Use 'cuda' se tiver GPU
                encode_kwargs={'normalize_embeddings': True}  # Melhora similaridade coseno
            )
            
            print("✅ Embeddings carregados")
        
        return self._embeddings
    
    @property
    def llm(self):
        """
        Lazy loading: só inicializa LLM quando necessário.
        """
        if self._llm is None:
            if not os.getenv("GROQ_API_KEY"):
                raise ValueError("GROQ_API_KEY não configurada no ambiente")
            
            self._llm = ChatGroq(
                model=self.llm_model_name,
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"]
            )
            
            print(f"✅ LLM inicializado: {self.llm_model_name}")
        
        return self._llm
    
    def create_vectorstore(
        self, 
        documents: List[Document],
        collection_name: str = None
    ) -> Chroma:
        """
        Cria um banco vetorial a partir de documentos.
        
        Este é o processo de INDEXAÇÃO:
        1. Pega cada chunk de texto
        2. Converte em vetor numérico (embedding)
        3. Salva no ChromaDB com metadados
        
        Args:
            documents: Lista de chunks processados
            collection_name: Nome da coleção no ChromaDB
            
        Returns:
            Instância do ChromaDB
        """
        if not documents:
            raise ValueError("Lista de documentos vazia")
        
        coll_name = collection_name or self.collection_name
        
        print(f"⏳ Criando banco vetorial com {len(documents)} chunks...")
        print(f"   📁 Salvando em: {self.persist_dir}")
        print(f"   📂 Collection: {coll_name}")
        
        # Cria o banco vetorial
        # Isso vai:
        # 1. Gerar embeddings para cada chunk (pode demorar!)
        # 2. Salvar no disco (persist_directory)
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=coll_name
        )
        
        print(f"✅ Banco vetorial criado: {vectorstore._collection.count()} vetores")
        
        self._vectorstore = vectorstore
        return vectorstore
    
    def load_vectorstore(self, collection_name: str = None) -> Chroma:
        """
        Carrega um banco vetorial existente do disco.
        
        Args:
            collection_name: Nome da coleção
            
        Returns:
            Instância do ChromaDB
        """
        if not Path(self.persist_dir).exists():
            raise FileNotFoundError(f"Banco vetorial não encontrado em: {self.persist_dir}")
        
        coll_name = collection_name or self.collection_name
        print(f"⏳ Carregando banco vetorial de: {self.persist_dir}")
        print(f"   📂 Collection: {coll_name}")
        
        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=coll_name   
        )
        
        print(f"✅ Banco carregado: {vectorstore._collection.count()} vetores")
        
        self._vectorstore = vectorstore
        return vectorstore
    
    def create_retriever(self, k: int = None, search_type: str = "similarity"):
        """
        Cria um recuperador (retriever) a partir do vectorstore.
        
        O retriever é o componente que busca os chunks mais relevantes.
        
        Args:
            k: Número de chunks a retornar (padrão: config.py)
            search_type: Tipo de busca ("similarity" ou "mmr")
                - similarity: Busca por similaridade coseno simples
                - mmr: Maximum Marginal Relevance (evita redundância)
        
        Returns:
            Retriever configurado
        """
        if self._vectorstore is None:
            raise ValueError("Vectorstore não inicializado. Chame create_vectorstore() primeiro.")
        
        k = k or RETRIEVAL_CONFIG["k"]
        
        print(f"⏳ Criando retriever (k={k}, tipo={search_type})...")
        
        self._retriever = self._vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        
        print("✅ Retriever criado")
        return self._retriever
    
    def format_documents(self, docs: List[Document]) -> str:
        """
        Formata documentos recuperados para inclusão no prompt.
        
        Cada chunk vira:
        [arquivo.pdf - p.3 - chunk 5]
        "Conteúdo do chunk aqui..."
        
        Args:
            docs: Lista de documentos recuperados
            
        Returns:
            String formatada para o prompt
        """
        formatted = []
        
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            source = meta.get("source_file", "desconhecido")
            page = meta.get("page", "?")
            chunk_idx = meta.get("chunk_index", "?")
            
            # Trunca conteúdo muito longo (opcional)
            content = doc.page_content[:1000]
            if len(doc.page_content) > 1000:
                content += "..."
            
            formatted.append(
                f"**[{source} - p.{page} - chunk {chunk_idx}]**\n{content}"
            )
        
        return "\n\n---\n\n".join(formatted)
    
    def create_rag_chain(self):
        """
        Cria o pipeline RAG completo.
        
        Pipeline: Pergunta → Busca no Vectorstore → Formata Contexto → LLM → Resposta
        
        Returns:
            Chain executável
        """
        if self._retriever is None:
            raise ValueError("Retriever não criado. Chame create_retriever() primeiro.")
        
        # Define o prompt do sistema
        system_prompt = """Você é um assistente acadêmico especializado em análise de papers científicos.

Sua tarefa é responder perguntas baseando-se ESTRITAMENTE no contexto fornecido dos papers.

Diretrizes:
1. **Cite as fontes**: Sempre mencione de qual paper veio cada informação (ex: "Segundo Silva et al. (2024)...")
2. **Seja preciso**: Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos papers fornecidos"
3. **Estruture bem**: Use seções como "Resumo", "Detalhes", "Limitações" quando apropriado
4. **Compare quando pedido**: Se perguntarem sobre diferenças entre estudos, faça comparação direta
5. **Linguagem acadêmica**: Use terminologia técnica apropriada, mas seja claro

Contexto dos Papers:
{context}"""

        # Cria o template de prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            ("human", "{question}")
        ])
        
        # Cria o pipeline RAG
        # RunnableParallel executa em paralelo:
        # - context: busca + formatação dos documentos
        # - question: apenas passa a pergunta adiante
        rag_chain = (
            RunnableParallel(
                context=self._retriever | self.format_documents,
                question=RunnablePassthrough()
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✅ Pipeline RAG criado")
        return rag_chain
    
    def query(
    self,
    question: str, 
    return_sources: bool = True
) -> Dict[str, Any]:
        if self._retriever is None:
            raise ValueError("Sistema RAG não inicializado completamente")
    
        print(f"⏳ Processando pergunta: {question[:50]}...")
    
        # 1. Busca documentos relevantes
        retrieved_docs = self._retriever.invoke(question)
        
        print(f"🐛 DEBUG - Documentos recuperados: {len(retrieved_docs)}")
        
        # 2. Formata o contexto
        context = self.format_documents(retrieved_docs)
        
        print(f"🐛 DEBUG - Tamanho do contexto: {len(context)} chars")
        
        # 3. Define o prompt (inline, não separado)
        system_prompt = """Você é um assistente acadêmico especializado em análise de papers científicos.

    Sua tarefa é responder perguntas baseando-se ESTRITAMENTE no contexto fornecido dos papers.

    Diretrizes:
    1. **Cite as fontes**: Sempre mencione de qual paper veio cada informação (ex: "Segundo o documento...")
    2. **Seja preciso**: Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos papers fornecidos"
    3. **Estruture bem**: Use seções como "Resumo", "Detalhes" quando apropriado
    4. **Linguagem acadêmica**: Use terminologia técnica apropriada, mas seja claro

    Contexto dos Papers:
    {context}

    Pergunta: {question}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt)
        ])
        
        # 4. Cria chain simples
        chain = prompt | self.llm | StrOutputParser()
        
        # 5. Executa chain com contexto e pergunta
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        # 6. Prepara resultado
        result = {
            "answer": answer,
            "metadata": {
                "chunks_retrieved": len(retrieved_docs),
                "model": self.llm_model_name,
                "embedding_model": self.embedding_model_name
            }
        }
        
        if return_sources:
            result["sources"] = retrieved_docs
        
        print("✅ Resposta gerada")
        return result

    def query_with_filters(
        self,
        question: str,
        author: str = None,
        year: int = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Faz uma pergunta ao sistema RAG com filtros de metadados.
        
        Args:
            question: Pergunta do usuário
            author: Filtrar por autor (opcional)
            year: Filtrar por ano (opcional)
            return_sources: Se True, retorna os chunks usados
            
        Returns:
            Dicionário com resposta e metadados
        """
        if self._vectorstore is None:
            raise ValueError("Vectorstore não inicializado")
        
        print(f"⏳ Processando pergunta com filtros...")
        if author:
            print(f"   👤 Filtro de autor: {author}")
        if year:
            print(f"   📅 Filtro de ano: {year}")
        
        # Constrói filtro para ChromaDB
        filter_dict = {}
        if author:
            filter_dict["author"] = author
        if year:
            filter_dict["year"] = year
        
        # Busca com filtros
        if filter_dict:
            # Usa similarity_search com filtro
            retrieved_docs = self._vectorstore.similarity_search(
                query=question,
                k=RETRIEVAL_CONFIG["k"],
                filter=filter_dict
            )
        else:
            # Busca normal sem filtros
            retrieved_docs = self._retriever.invoke(question)
        
        print(f"🐛 DEBUG - Documentos recuperados (com filtros): {len(retrieved_docs)}")
        
        if not retrieved_docs:
            return {
                "answer": f"Não encontrei documentos correspondentes aos filtros especificados (autor: {author}, ano: {year}).",
                "sources": [],
                "metadata": {
                    "chunks_retrieved": 0,
                    "filters_applied": filter_dict
                }
            }
        
        # Formata contexto
        context = self.format_documents(retrieved_docs)
        
        # Define prompt
        system_prompt = """Você é um assistente acadêmico especializado em análise de papers científicos.

    Sua tarefa é responder perguntas baseando-se ESTRITAMENTE no contexto fornecido dos papers.

    Diretrizes:
    1. **Cite as fontes**: Sempre mencione de qual paper veio cada informação
    2. **Seja preciso**: Se a resposta não estiver no contexto, diga claramente
    3. **Estruture bem**: Use seções quando apropriado
    4. **Linguagem acadêmica**: Use terminologia técnica, mas seja claro

    Contexto dos Papers:
    {context}

    Pergunta: {question}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt)
        ])
        
        # Cria chain
        chain = prompt | self.llm | StrOutputParser()
        
        # Executa
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        # Resultado
        result = {
            "answer": answer,
            "metadata": {
                "chunks_retrieved": len(retrieved_docs),
                "filters_applied": filter_dict,
                "model": self.llm_model_name,
                "embedding_model": self.embedding_model_name
            }
        }
        
        if return_sources:
            result["sources"] = retrieved_docs
        
        print("✅ Resposta gerada")
        return result

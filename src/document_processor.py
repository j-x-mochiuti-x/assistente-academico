"""
Módulo responsável por processar documentos acadêmicos (PDFs).
Extrai texto, limpa, divide em chunks e prepara para vetorização.
"""
import os
import tempfile
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_CONFIG

class DocumentProcessor:
    """
    Classe para processar docs acadêmicos.
    
    Responsabilidades da Classe:
    1. Carregar PDFs
    2. Extrair Metadados (título, autor, ano)
    3. Limpar textos
    4. Dividir em CHUNKS semânticos"""

    def __init__(self, chunk_size: int=None, chunk_overlap: int=None):
        """Inicializa o processador.
        
        Args:
            chunk_size: Tamanho dos chunks (padrão do config.py)
            chunk_overlap: Sobreposição entre chunks (padrão do config.py)"""
        
        self.chunk_size = chunk_size or CHUNK_CONFIG["chunk_size"]
        self.chunk_overlap = chunk_overlap or CHUNK_CONFIG["chunk_overlap"]
        # Inicializa o divisor de texto
        # RecursiveCharacterTextSplitter tenta dividir por parágrafos primeiro,
        # depois por sentenças, mantendo contexto semântico
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # Ordem de preferência
            length_function=len
        )
    
    def load_pdf(self, pdf_source: Any, filename: str="documento.pdf") -> List[Document]:
        """
        Carrega um PDF e extrai o conteúdo de dentro.

        Args:
            pdf_source: Pode ser bytes (upload) ou caminho de arquivo
            filename: Nome do arquivo (para metadados)

        Returns:
            lista de docs LangChain (1/pagina)
        
        Raises:
            ValueError: Caso o PDF estiver vazio ou corrompido
        """
        #Caso 1: S receber bytes (upload do Streamlit)
        if isinstance(pdf_source, bytes):
            return self._load_from_bytes(pdf_source, filename)
        
        #Caso2: se receber um objeto de arquivo do Streamlit
        elif hasattr(pdf_source, 'read'):
            pdf_bytes = pdf_source.read()
            return self._load_from_bytes(pdf_bytes, filename)
        
        #caso3: se recever um caminho de arquivo
        elif isinstance(pdf_source, (str, Path)):
            if not os.path.exists(pdf_source):
                raise FileNotFoundError(f"Arquivo não encontrado: {pdf_source}")
            return self._load_from_path(pdf_source)
        
        else:
            raise TypeError(f"Tipo de entrada não suportada: {type(pdf_source)}")
        

    def _load_from_bytes(self, pdf_bytes: bytes, filename: str) -> List[Document]:
        """
        Carrega PDF a partir de bytes (método interno).
        Cria arquivo temporário, processa e limpa
        """
        # cria arquivo temp
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name

        try:
            #carrega do PDF
            docs = self._load_from_path(tmp_path)

            #adiciona nome do' arquivo ao metadado
            for doc in docs:
                doc.metadata["source_file"] = filename

            return docs
        
        finally:
            #sempre limpa o arquivo temporário (memso se der erro)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _load_from_path(self, file_path: str) -> List[Document]:
        """
        Carrega PDF a partir de um caminho (método interno).
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            if not documents:
                raise ValueError("PDF vazio ou não legivel")
            
            return documents
        
        except Exception as e:
            raise ValueError(f"Erro ao processar PDF:{str(e)}")
        
    def clean_text(self, text:str) -> str:
        """
        Limpa o texto extraído do PDF.

        REMOVE:
        1. Multiplos espaços em branco
        2. Quebras de linha excessivas
        3. Caracteres especiais problemáticos

        Args:
        text: texto bruto

        Returns:
        texto limpo
        """
        import re

        #remove multiplos espaços
        text = re.sun(r' +', ' ', text)
        
        #remove espaços no inicio e fim de cada linha
        text = re.sub(r'\n{3,}', '\n\n', text)

        #remove espaços no inicio e fim de cada linha
        text = '\n'.join(line.strip() for line in text.split('\n'))

        #remove hifens de quebra de linha (ex: "reco-\nmmendation" -> "recommendation")
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

        return text.strip()
    
    def extract_metadata(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Extrai metadados úteis dos documentos.
        
        ⚠️ AVISO DE INCERTEZA: A extração de título/autor é heurística
        e pode não funcionar perfeitamente para todos os formatos de paper.
        
        Args:
            documents: Lista de documentos carregados
            
        Returns:
            Dicionário com metadados extraídos
        """

        if not documents:
            return {}
    
        # Pega a primeira página (geralmente tem título e autores)
        first_page = documents[0].page_content

        metadata = {
            "total_pages": len(documents),
            "source_file": documents[0].metadata.get("source_file", "desconhecido"),
        }

         # Tenta extrair título (geralmente está no início, em maiúsculas ou negrito)
        # NOTA: Isso é uma heurística simples, não é 100% confiável
        lines =  first_page.split('\n')
        for i, line in enumerate(lines[:10]):
            if len(line) > 20 and line.strip():
                metadata["possible_title"] = line.strip()
                break
        return metadata
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Divide documentos em chunks menores.
        
        Args:
            documents: Lista de documentos a dividir
            
        Returns:
            Lista de chunks (documentos menores)
        """
        # Limpa o texto de cada documento antes de dividir
        cleaned_docs = []
        for doc in documents:
            cleaned_content = self.clean_text(doc.page_content)
            cleaned_doc = Document(
                page_content=cleaned_content,
                metadata=doc.metadata
            )
            cleaned_docs.append(cleaned_doc)

        #divide em chunks
        chunks = self.text_splitter.split_documents(cleaned_docs)
        
        #adiciona indice de chunk aos metadados
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        
        return chunks
    
    def process_pdf(self, pdf_source: Any, filename: str = "docuemnto.pdf") -> Dict[str, Any]:
        """
        Pipeline completo de processamento de PDF.
        
        Este é o método principal que você vai chamar.
        Faz tudo: carrega, extrai metadados, limpa e divide.
        
        Args:
            pdf_source: PDF (bytes, arquivo ou caminho)
            filename: Nome do arquivo
            
        Returns:
            Dicionário contendo:
            1. documents: Lista de chunks processados
            2. metadata: Metadados extraídos
            3. stats: Estatísticas do processamento
        """
        try:
            #carrega o pdf
            raw_documents = self.load_pdf(pdf_source, filename)
            #extrai metadados
            metadata = self.extract_metadata(raw_documents)
            #divide em chunks
            chunks = self.split_documents(raw_documents)
            #calcula estatisticas
            stats = {
                "total_pages": len(raw_documents),
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0,
            }

            return{
                "documents": chunks,
                "metadata": metadata,
                "stats": stats,
                "success": True,
                "error": None
            }
        
        except Exception as e:
            # Retorna erro estruturado em vez de crashar
            return{
                 "documents": [],
                "metadata": {},
                "stats": {},
                "success": False,
                "error": str(e)
            }
        
#Função auxiliar para uso direto (sem instanciar classe)
def process_single_pdf(pdf_source: Any, filename: str = "documento.pdf") -> Dict[str, Any]:
    """
    Função de conveniência para processar um único PDF.

    Usage:
    result = process_single_pdf(uploaded_file, "paper.pdf)
    if result["sucess"]:
        chunks = result["documents"]
    """
    processor = DocumentProcessor()
    return processor.process_pdf(pdf_source, filename)
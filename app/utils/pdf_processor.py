import logging
from typing import List, Optional, Tuple
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class PDFProcessor:
    """
    Utility class for processing PDF files and extracting text content.
    """
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {str(e)}")
            raise

    @staticmethod
    def split_text(text: str, chunk_size: int = 5000, chunk_overlap: int = 50) -> List[Document]:
        """
        Split text into manageable chunks for processing by LLMs.
        
        Args:
            text: Text content to split
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List of Document objects containing split text
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.create_documents([text])

    @staticmethod
    def get_paper_metadata(file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Attempt to extract metadata from the PDF such as title and authors.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (title, authors) if available, or (None, None)
        """
        try:
            reader = PdfReader(file_path)
            metadata = reader.metadata
            title = metadata.get('/Title', None)
            author = metadata.get('/Author', None)
            
            return title, author
        except Exception as e:
            logging.error(f"Error extracting metadata from PDF: {str(e)}")
            return None, None

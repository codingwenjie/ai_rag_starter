"""文档加载器模块

支持多种文档格式的加载和预处理，包括PDF、Markdown、HTML、TXT等格式。
提供统一的文档加载接口和元数据提取功能。
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import mimetypes
from datetime import datetime

try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        UnstructuredMarkdownLoader,
        UnstructuredHTMLLoader,
        DirectoryLoader
    )
    from langchain.schema import Document
except ImportError:
    # 如果LangChain未安装，提供基础实现
    class Document:
        def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

logger = logging.getLogger(__name__)

class DocumentLoader:
    """统一文档加载器
    
    支持多种文档格式的加载，提供统一的接口和元数据处理。
    """
    
    def __init__(self):
        """初始化文档加载器"""
        self.supported_extensions = {
            '.pdf': self._load_pdf,
            '.txt': self._load_text,
            '.md': self._load_markdown,
            '.markdown': self._load_markdown,
            '.html': self._load_html,
            '.htm': self._load_html,
        }
        
    def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """加载单个文档
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            List[Document]: 文档对象列表
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        extension = file_path.suffix.lower()
        
        if extension not in self.supported_extensions:
            raise ValueError(f"不支持的文件格式: {extension}")
            
        try:
            loader_func = self.supported_extensions[extension]
            documents = loader_func(file_path)
            
            # 添加通用元数据
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_name': file_path.name,
                    'file_size': file_path.stat().st_size,
                    'file_type': extension,
                    'load_time': datetime.now().isoformat(),
                })
                
            logger.info(f"成功加载文档: {file_path}, 共 {len(documents)} 个片段")
            return documents
            
        except Exception as e:
            logger.error(f"加载文档失败: {file_path}, 错误: {str(e)}")
            raise
            
    def load_directory(self, 
                      directory_path: Union[str, Path],
                      recursive: bool = True,
                      file_filter: Optional[str] = None) -> List[Document]:
        """加载目录中的所有文档
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归加载子目录
            file_filter: 文件过滤模式（如 '*.pdf'）
            
        Returns:
            List[Document]: 所有文档对象列表
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"目录不存在或不是有效目录: {directory_path}")
            
        all_documents = []
        
        # 获取文件列表
        if recursive:
            pattern = '**/*' if not file_filter else f'**/{file_filter}'
            files = directory_path.glob(pattern)
        else:
            pattern = '*' if not file_filter else file_filter
            files = directory_path.glob(pattern)
            
        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    documents = self.load_document(file_path)
                    all_documents.extend(documents)
                except Exception as e:
                    logger.warning(f"跳过文件 {file_path}: {str(e)}")
                    
        logger.info(f"目录加载完成: {directory_path}, 共 {len(all_documents)} 个文档片段")
        return all_documents
        
    def _load_pdf(self, file_path: Path) -> List[Document]:
        """加载PDF文档
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            List[Document]: 文档对象列表
        """
        try:
            # 尝试使用LangChain的PyPDFLoader
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # 添加页码信息
            for i, doc in enumerate(documents):
                doc.metadata['page'] = i + 1
                
            return documents
            
        except (ImportError, NameError):
            # 如果LangChain不可用，使用基础PDF处理
            try:
                import PyPDF2
                
                documents = []
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text.strip():
                            doc = Document(
                                page_content=text,
                                metadata={'page': page_num + 1}
                            )
                            documents.append(doc)
                            
                return documents
                
            except ImportError:
                raise ImportError("需要安装 PyPDF2 或 langchain 来处理PDF文件")
                
    def _load_text(self, file_path: Path) -> List[Document]:
        """加载文本文档
        
        Args:
            file_path: 文本文件路径
            
        Returns:
            List[Document]: 文档对象列表
        """
        try:
            # 尝试使用LangChain的TextLoader
            loader = TextLoader(str(file_path), encoding='utf-8')
            return loader.load()
            
        except (ImportError, NameError):
            # 基础文本加载实现
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            return [Document(page_content=content, metadata={})]
            
    def _load_markdown(self, file_path: Path) -> List[Document]:
        """加载Markdown文档
        
        Args:
            file_path: Markdown文件路径
            
        Returns:
            List[Document]: 文档对象列表
        """
        try:
            # 尝试使用LangChain的UnstructuredMarkdownLoader
            loader = UnstructuredMarkdownLoader(str(file_path))
            return loader.load()
            
        except (ImportError, NameError):
            # 基础Markdown加载实现
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            return [Document(page_content=content, metadata={'format': 'markdown'})]
            
    def _load_html(self, file_path: Path) -> List[Document]:
        """加载HTML文档
        
        Args:
            file_path: HTML文件路径
            
        Returns:
            List[Document]: 文档对象列表
        """
        try:
            # 尝试使用LangChain的UnstructuredHTMLLoader
            loader = UnstructuredHTMLLoader(str(file_path))
            return loader.load()
            
        except (ImportError, NameError):
            # 基础HTML加载实现
            try:
                from bs4 import BeautifulSoup
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()
                    
                soup = BeautifulSoup(html_content, 'html.parser')
                text_content = soup.get_text(separator='\n', strip=True)
                
                return [Document(page_content=text_content, metadata={'format': 'html'})]
                
            except ImportError:
                # 如果BeautifulSoup不可用，直接读取HTML
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                return [Document(page_content=content, metadata={'format': 'html'})]
                
    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式列表
        
        Returns:
            List[str]: 支持的文件扩展名列表
        """
        return list(self.supported_extensions.keys())
        
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """验证文件是否可以加载
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        file_path = Path(file_path)
        
        result = {
            'valid': False,
            'exists': file_path.exists(),
            'is_file': file_path.is_file() if file_path.exists() else False,
            'extension': file_path.suffix.lower(),
            'supported': False,
            'size': None,
            'error': None
        }
        
        try:
            if not result['exists']:
                result['error'] = '文件不存在'
                return result
                
            if not result['is_file']:
                result['error'] = '不是有效文件'
                return result
                
            result['supported'] = result['extension'] in self.supported_extensions
            
            if not result['supported']:
                result['error'] = f"不支持的文件格式: {result['extension']}"
                return result
                
            result['size'] = file_path.stat().st_size
            result['valid'] = True
            
        except Exception as e:
            result['error'] = str(e)
            
        return result


class DocumentMetadataExtractor:
    """文档元数据提取器
    
    提取文档的各种元数据信息，如标题、作者、创建时间等。
    """
    
    @staticmethod
    def extract_metadata(document: Document, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """提取文档元数据
        
        Args:
            document: 文档对象
            file_path: 文件路径（可选）
            
        Returns:
            Dict[str, Any]: 元数据字典
        """
        metadata = document.metadata.copy()
        
        # 提取文本统计信息
        content = document.page_content
        metadata.update({
            'char_count': len(content),
            'word_count': len(content.split()),
            'line_count': len(content.splitlines()),
        })
        
        # 如果有文件路径，提取文件信息
        if file_path:
            file_path = Path(file_path)
            if file_path.exists():
                stat = file_path.stat()
                metadata.update({
                    'file_size': stat.st_size,
                    'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
                
        # 尝试提取内容特征
        metadata.update(DocumentMetadataExtractor._extract_content_features(content))
        
        return metadata
        
    @staticmethod
    def _extract_content_features(content: str) -> Dict[str, Any]:
        """提取内容特征
        
        Args:
            content: 文档内容
            
        Returns:
            Dict[str, Any]: 内容特征字典
        """
        features = {}
        
        # 检测语言（简单实现）
        chinese_chars = sum(1 for char in content if '\u4e00' <= char <= '\u9fff')
        english_chars = sum(1 for char in content if char.isalpha() and ord(char) < 128)
        
        if chinese_chars > english_chars:
            features['language'] = 'zh'
        elif english_chars > 0:
            features['language'] = 'en'
        else:
            features['language'] = 'unknown'
            
        # 检测内容类型
        if '```' in content or 'def ' in content or 'class ' in content:
            features['content_type'] = 'code'
        elif content.count('#') > 3:  # 多个标题
            features['content_type'] = 'documentation'
        elif content.count('.') > len(content) / 100:  # 句号密度高
            features['content_type'] = 'article'
        else:
            features['content_type'] = 'text'
            
        return features


# 使用示例
if __name__ == "__main__":
    # 创建文档加载器
    loader = DocumentLoader()
    
    # 加载单个文档
    try:
        documents = loader.load_document("example.pdf")
        print(f"加载了 {len(documents)} 个文档片段")
        
        for i, doc in enumerate(documents[:2]):  # 显示前两个片段
            print(f"\n片段 {i+1}:")
            print(f"内容长度: {len(doc.page_content)}")
            print(f"元数据: {doc.metadata}")
            print(f"内容预览: {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"加载失败: {e}")
        
    # 验证文件
    validation = loader.validate_file("example.pdf")
    print(f"\n文件验证结果: {validation}")
    
    # 显示支持的格式
    print(f"\n支持的文件格式: {loader.get_supported_formats()}")
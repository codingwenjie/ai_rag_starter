"""文本切分器模块

实现多种文本切分策略，将长文档切分为合适的块，保持语义完整性。
支持基于字符数、语义和递归的切分方法。
"""

import re
import logging
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod

try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
        TokenTextSplitter
    )
    from langchain.schema import Document
except ImportError:
    # 如果LangChain未安装，提供基础实现
    class Document:
        def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

logger = logging.getLogger(__name__)

class BaseTextSplitter(ABC):
    """文本切分器基类"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 length_function: Callable[[str], int] = len,
                 keep_separator: bool = False):
        """初始化文本切分器
        
        Args:
            chunk_size: 每个块的最大大小
            chunk_overlap: 块之间的重叠大小
            length_function: 计算文本长度的函数
            keep_separator: 是否保留分隔符
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.keep_separator = keep_separator
        
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """切分文本
        
        Args:
            text: 要切分的文本
            
        Returns:
            List[str]: 切分后的文本块列表
        """
        pass
        
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档列表
        
        Args:
            documents: 文档列表
            
        Returns:
            List[Document]: 切分后的文档列表
        """
        split_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                # 创建新的文档对象
                new_metadata = doc.metadata.copy()
                new_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk),
                    'original_doc_id': id(doc)
                })
                
                split_doc = Document(
                    page_content=chunk,
                    metadata=new_metadata
                )
                split_docs.append(split_doc)
                
        logger.info(f"文档切分完成: {len(documents)} -> {len(split_docs)} 个块")
        return split_docs
        
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """合并切分的文本块
        
        Args:
            splits: 切分的文本列表
            separator: 分隔符
            
        Returns:
            List[str]: 合并后的文本块列表
        """
        separator_len = self.length_function(separator)
        
        docs = []
        current_doc = []
        total = 0
        
        for split in splits:
            split_len = self.length_function(split)
            
            # 如果当前块加上新的split超过了chunk_size
            if total + split_len + (separator_len if current_doc else 0) > self.chunk_size:
                if current_doc:
                    # 保存当前块
                    doc = separator.join(current_doc)
                    if doc:
                        docs.append(doc)
                        
                    # 处理重叠
                    while (total > self.chunk_overlap and 
                           len(current_doc) > 0 and 
                           total + split_len + separator_len > self.chunk_size):
                        total -= self.length_function(current_doc[0]) + separator_len
                        current_doc = current_doc[1:]
                        
            current_doc.append(split)
            total += split_len + (separator_len if len(current_doc) > 1 else 0)
            
        # 处理最后一个块
        if current_doc:
            doc = separator.join(current_doc)
            if doc:
                docs.append(doc)
                
        return docs


class CharacterTextSplitter(BaseTextSplitter):
    """基于字符的文本切分器"""
    
    def __init__(self, 
                 separator: str = "\n\n",
                 **kwargs):
        """初始化字符文本切分器
        
        Args:
            separator: 分隔符
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.separator = separator
        
    def split_text(self, text: str) -> List[str]:
        """基于字符切分文本
        
        Args:
            text: 要切分的文本
            
        Returns:
            List[str]: 切分后的文本块列表
        """
        if self.separator:
            splits = text.split(self.separator)
        else:
            splits = list(text)
            
        return self._merge_splits(splits, self.separator)


class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """递归字符文本切分器
    
    按照优先级顺序尝试不同的分隔符进行切分。
    """
    
    def __init__(self, 
                 separators: Optional[List[str]] = None,
                 **kwargs):
        """初始化递归字符文本切分器
        
        Args:
            separators: 分隔符列表，按优先级排序
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
    def split_text(self, text: str) -> List[str]:
        """递归切分文本
        
        Args:
            text: 要切分的文本
            
        Returns:
            List[str]: 切分后的文本块列表
        """
        return self._split_text_recursive(text, self.separators)
        
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """递归切分文本的内部实现
        
        Args:
            text: 要切分的文本
            separators: 分隔符列表
            
        Returns:
            List[str]: 切分后的文本块列表
        """
        final_chunks = []
        
        # 获取当前分隔符
        separator = separators[-1] if separators else ""
        new_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if re.search(re.escape(sep), text):
                separator = sep
                new_separators = separators[i + 1:]
                break
                
        # 使用当前分隔符切分
        splits = text.split(separator) if separator else [text]
        
        # 处理每个切分结果
        good_splits = []
        for split in splits:
            if self.length_function(split) < self.chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    merged_text = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                    
                if not new_separators:
                    # 如果没有更多分隔符，强制切分
                    final_chunks.extend(self._force_split(split))
                else:
                    # 递归使用下一个分隔符
                    other_info = self._split_text_recursive(split, new_separators)
                    final_chunks.extend(other_info)
                    
        if good_splits:
            merged_text = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged_text)
            
        return final_chunks
        
    def _force_split(self, text: str) -> List[str]:
        """强制切分文本（当无法使用分隔符时）
        
        Args:
            text: 要切分的文本
            
        Returns:
            List[str]: 切分后的文本块列表
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 如果不是最后一块，尝试在单词边界切分
            if end < len(text):
                # 向后查找空格
                while end > start and text[end] not in [' ', '\n', '\t']:
                    end -= 1
                    
                # 如果找不到合适的边界，就强制切分
                if end == start:
                    end = start + self.chunk_size
                    
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            # 计算下一个开始位置（考虑重叠）
            start = max(start + 1, end - self.chunk_overlap)
            
        return chunks


class SemanticTextSplitter(BaseTextSplitter):
    """基于语义的文本切分器
    
    尝试在语义边界处切分文本，如段落、句子等。
    """
    
    def __init__(self, **kwargs):
        """初始化语义文本切分器"""
        super().__init__(**kwargs)
        
        # 句子结束标记
        self.sentence_endings = r'[.!?。！？]\s+'
        # 段落分隔符
        self.paragraph_separators = r'\n\s*\n'
        
    def split_text(self, text: str) -> List[str]:
        """基于语义切分文本
        
        Args:
            text: 要切分的文本
            
        Returns:
            List[str]: 切分后的文本块列表
        """
        # 首先按段落切分
        paragraphs = re.split(self.paragraph_separators, text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # 如果当前段落加上现有块不超过限制，直接添加
            if (self.length_function(current_chunk + paragraph) <= self.chunk_size or 
                not current_chunk):
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk)
                    
                # 如果段落本身太长，需要进一步切分
                if self.length_function(paragraph) > self.chunk_size:
                    sentence_chunks = self._split_by_sentences(paragraph)
                    chunks.extend(sentence_chunks[:-1])  # 除了最后一个
                    current_chunk = sentence_chunks[-1] if sentence_chunks else ""
                else:
                    current_chunk = paragraph
                    
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
        
    def _split_by_sentences(self, text: str) -> List[str]:
        """按句子切分文本
        
        Args:
            text: 要切分的文本
            
        Returns:
            List[str]: 切分后的句子块列表
        """
        # 按句子切分
        sentences = re.split(self.sentence_endings, text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 重新添加句号（简单处理）
            if not sentence.endswith(('.', '!', '?', '。', '！', '？')):
                sentence += '.'
                
            if (self.length_function(current_chunk + sentence) <= self.chunk_size or 
                not current_chunk):
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks


class MarkdownTextSplitter(BaseTextSplitter):
    """Markdown文档切分器
    
    专门用于切分Markdown格式的文档，保持标题结构。
    """
    
    def __init__(self, **kwargs):
        """初始化Markdown文本切分器"""
        super().__init__(**kwargs)
        
        # Markdown标题模式
        self.header_pattern = r'^(#{1,6})\s+(.+)$'
        
    def split_text(self, text: str) -> List[str]:
        """切分Markdown文本
        
        Args:
            text: 要切分的Markdown文本
            
        Returns:
            List[str]: 切分后的文本块列表
        """
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_headers = []  # 保存当前的标题层次
        
        for line in lines:
            header_match = re.match(self.header_pattern, line)
            
            if header_match:
                # 遇到新标题
                level = len(header_match.group(1))
                title = header_match.group(2)
                
                # 如果当前块太大，保存它
                if (current_chunk and 
                    self.length_function('\n'.join(current_chunk)) > self.chunk_size):
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    
                # 更新标题层次
                current_headers = current_headers[:level-1] + [title]
                
                # 添加标题上下文
                header_context = self._build_header_context(current_headers)
                if header_context:
                    current_chunk.append(header_context)
                    
            current_chunk.append(line)
            
            # 检查是否需要切分
            if self.length_function('\n'.join(current_chunk)) > self.chunk_size:
                # 尝试在合适的位置切分
                split_point = self._find_split_point(current_chunk)
                if split_point > 0:
                    chunks.append('\n'.join(current_chunk[:split_point]))
                    current_chunk = current_chunk[split_point:]
                    
        # 添加最后一个块
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return [chunk for chunk in chunks if chunk.strip()]
        
    def _build_header_context(self, headers: List[str]) -> str:
        """构建标题上下文
        
        Args:
            headers: 标题列表
            
        Returns:
            str: 标题上下文字符串
        """
        if not headers:
            return ""
            
        context_parts = []
        for i, header in enumerate(headers):
            prefix = "#" * (i + 1)
            context_parts.append(f"{prefix} {header}")
            
        return "\n".join(context_parts)
        
    def _find_split_point(self, lines: List[str]) -> int:
        """找到合适的切分点
        
        Args:
            lines: 文本行列表
            
        Returns:
            int: 切分点索引
        """
        # 从后往前找，寻找空行或标题
        for i in range(len(lines) - 1, 0, -1):
            if not lines[i].strip():  # 空行
                return i
            if re.match(self.header_pattern, lines[i]):  # 标题
                return i
                
        # 如果找不到合适的切分点，返回中间位置
        return len(lines) // 2


class SmartTextSplitter:
    """智能文本切分器
    
    根据文档类型自动选择合适的切分策略。
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """初始化智能文本切分器
        
        Args:
            chunk_size: 每个块的最大大小
            chunk_overlap: 块之间的重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化各种切分器
        self.splitters = {
            'markdown': MarkdownTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            ),
            'semantic': SemanticTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            ),
            'recursive': RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            ),
            'character': CharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
        }
        
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """智能切分文档
        
        Args:
            documents: 文档列表
            
        Returns:
            List[Document]: 切分后的文档列表
        """
        all_split_docs = []
        
        for doc in documents:
            # 根据文档类型选择切分器
            splitter_type = self._detect_document_type(doc)
            splitter = self.splitters[splitter_type]
            
            # 切分文档
            split_docs = splitter.split_documents([doc])
            
            # 添加切分器类型信息
            for split_doc in split_docs:
                split_doc.metadata['splitter_type'] = splitter_type
                
            all_split_docs.extend(split_docs)
            
        logger.info(f"智能切分完成: {len(documents)} -> {len(all_split_docs)} 个块")
        return all_split_docs
        
    def _detect_document_type(self, document: Document) -> str:
        """检测文档类型
        
        Args:
            document: 文档对象
            
        Returns:
            str: 文档类型
        """
        content = document.page_content
        metadata = document.metadata
        
        # 检查文件扩展名
        if metadata.get('format') == 'markdown' or metadata.get('file_type') in ['.md', '.markdown']:
            return 'markdown'
            
        # 检查内容特征
        if re.search(r'^#{1,6}\s+', content, re.MULTILINE):  # Markdown标题
            return 'markdown'
            
        # 检查是否有明显的段落结构
        if content.count('\n\n') > len(content) / 500:  # 段落密度
            return 'semantic'
            
        # 默认使用递归切分
        return 'recursive'


# 使用示例
if __name__ == "__main__":
    # 示例文本
    sample_text = """
    # 标题一
    
    这是第一段内容。这段内容包含了一些重要的信息，需要被正确地切分和处理。
    
    ## 子标题
    
    这是第二段内容。这段内容也很重要，包含了更多的细节信息。
    我们需要确保切分后的内容保持语义的完整性。
    
    ### 更小的标题
    
    这是第三段内容，包含了具体的实现细节和代码示例。
    """
    
    # 创建文档对象
    doc = Document(
        page_content=sample_text,
        metadata={'format': 'markdown'}
    )
    
    # 测试不同的切分器
    splitters = {
        '智能切分器': SmartTextSplitter(chunk_size=200, chunk_overlap=50),
        'Markdown切分器': MarkdownTextSplitter(chunk_size=200, chunk_overlap=50),
        '语义切分器': SemanticTextSplitter(chunk_size=200, chunk_overlap=50),
        '递归切分器': RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    }
    
    for name, splitter in splitters.items():
        print(f"\n=== {name} ===")
        if hasattr(splitter, 'split_documents'):
            chunks = splitter.split_documents([doc])
        else:
            chunks = splitter.split_documents([doc])
            
        print(f"切分结果: {len(chunks)} 个块")
        
        for i, chunk in enumerate(chunks[:2]):  # 只显示前两个块
            print(f"\n块 {i+1} (长度: {len(chunk.page_content)}):")
            print(chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content)
"""文件处理工具函数

包含文件名安全处理、文件类型验证等工具函数。
"""

import re
import os
from typing import Optional


def safe_filename(filename: str, max_length: int = 255) -> str:
    """生成安全的文件名
    
    Args:
        filename: 原始文件名
        max_length: 最大文件名长度
        
    Returns:
        str: 安全的文件名
    """
    if not filename:
        return "unnamed_file"
    
    # 获取文件名和扩展名
    name, ext = os.path.splitext(filename)
    
    # 移除或替换不安全的字符
    # 只保留字母、数字、下划线、连字符和点
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    safe_ext = re.sub(r'[^a-zA-Z0-9.]', '', ext)
    
    # 移除连续的下划线
    safe_name = re.sub(r'_+', '_', safe_name)
    
    # 移除开头和结尾的下划线和点
    safe_name = safe_name.strip('_.')
    
    # 如果处理后名称为空，使用默认名称
    if not safe_name:
        safe_name = "unnamed_file"
    
    # 组合文件名
    full_name = safe_name + safe_ext
    
    # 限制长度
    if len(full_name) > max_length:
        # 保留扩展名，截断主文件名
        available_length = max_length - len(safe_ext)
        if available_length > 0:
            safe_name = safe_name[:available_length]
            full_name = safe_name + safe_ext
        else:
            # 如果扩展名太长，只保留部分
            full_name = full_name[:max_length]
    
    return full_name


def get_file_extension(filename: str) -> Optional[str]:
    """获取文件扩展名
    
    Args:
        filename: 文件名
        
    Returns:
        Optional[str]: 文件扩展名（包含点），如果没有扩展名则返回None
    """
    if not filename:
        return None
        
    _, ext = os.path.splitext(filename.lower())
    return ext if ext else None


def is_text_file(filename: str) -> bool:
    """判断是否为文本文件
    
    Args:
        filename: 文件名
        
    Returns:
        bool: 是否为文本文件
    """
    text_extensions = {
        '.txt', '.md', '.html', '.htm', '.csv', 
        '.json', '.xml', '.yaml', '.yml', '.log'
    }
    
    ext = get_file_extension(filename)
    return ext in text_extensions if ext else False
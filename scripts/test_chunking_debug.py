#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试切分逻辑优化效果 - 调试版本"""

import sys
from pathlib import Path
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.indexing.markdown_chunker import split_markdown_chunks
from rag.retrieval.tokenization import estimate_token_size


# 测试用例 1：加密说明章节（应保持完整）
markdown1 = """
# API 文档

## 接口列表

这是接口列表的介绍。

## 加密说明

### DES 加密算法

系统使用 DES 算法对请求参数进行加密。

### 加密步骤

1. 将请求参数按字典序排序
2. 拼接成字符串：key1=value1&key2=value2
3. 使用 DES 算法加密
4. Base64 编码

### 请求头设置

加密后的数据需要放在 X-Crypto 请求头中：

```
X-Crypto: <encrypted_base64_string>
```

### 示例代码

```python
import base64
from Crypto.Cipher import DES

def encrypt_params(params, key):
    # 排序
    sorted_params = sorted(params.items())
    # 拼接
    data = "&".join([f"{k}={v}" for k, v in sorted_params])
    # 加密
    cipher = DES.new(key, DES.MODE_ECB)
    encrypted = cipher.encrypt(data)
    return base64.b64encode(encrypted).decode()
```

## 生成订单接口

POST /api/order/create

创建新的订单。
"""

print("=" * 80)
print("测试用例：加密说明章节")
print("=" * 80)

chunks = split_markdown_chunks(markdown1, chunk_size=512, chunk_overlap=100)

print(f"\n总 chunk 数: {len(chunks)}\n")

for i, chunk in enumerate(chunks, 1):
    tokens = estimate_token_size(chunk["text"])
    print(f"\n--- Chunk {i} ---")
    print(f"heading_path: {chunk['heading_path']}")
    print(f"tokens: {tokens}")
    print(f"content length: {len(chunk['text'])} chars")
    print(f"包含'加密'关键词: {'加密' in chunk['text']}")
    print(f"包含'DES': {'DES' in chunk['text']}")
    print(f"包含'示例代码': {'示例代码' in chunk['text']}")
    print(f"\nFull content:\n{chunk['text']}")
    print("-" * 80)

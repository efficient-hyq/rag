#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试切分逻辑优化效果"""

import sys
from pathlib import Path

# 修复 Windows 控制台编码问题
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.indexing.markdown_chunker import split_markdown_chunks
from rag.retrieval.tokenization import estimate_token_size


def test_atomic_section_chunking():
    """测试原子章节不被切分"""

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
    print("测试用例 1：加密说明章节（应保持完整）")
    print("=" * 80)

    chunks = split_markdown_chunks(markdown1, chunk_size=512, chunk_overlap=100)

    print(f"\n总 chunk 数: {len(chunks)}\n")

    encryption_chunks = [c for c in chunks if "加密" in c["heading_path"]]

    if encryption_chunks:
        print(f"包含'加密'的 chunk 数: {len(encryption_chunks)}")
        for i, chunk in enumerate(encryption_chunks, 1):
            tokens = estimate_token_size(chunk["text"])
            print(f"\n--- Chunk {i} ---")
            print(f"heading_path: {chunk['heading_path']}")
            print(f"tokens: {tokens}")
            print(f"content length: {len(chunk['text'])} chars")
            print(f"content preview (first 300 chars):\n{chunk['text'][:300]}...")
            print(f"content preview (last 300 chars):\n...{chunk['text'][-300:]}")

        if len(encryption_chunks) == 1:
            print("\n✅ 加密说明章节保持完整（未被切分）")
        else:
            print(f"\n⚠️  加密说明章节被切分成 {len(encryption_chunks)} 个 chunk")
    else:
        print("\n❌ 未找到加密说明相关 chunk")

    # 测试用例 2：错误码章节（应保持完整）
    markdown2 = """
# 错误码说明

## 通用错误码

| 错误码 | 说明 | 解决方案 |
|-------|------|---------|
| 401 | 未授权 | 检查 API Key 是否正确 |
| 403 | 禁止访问 | 检查权限配置 |
| 404 | 资源不存在 | 检查请求路径 |
| 500 | 服务器错误 | 联系技术支持 |

## 业务错误码

| 错误码 | 说明 | 解决方案 |
|-------|------|---------|
| 1001 | 订单不存在 | 检查订单号 |
| 1002 | 订单已支付 | 不能重复支付 |
| 1003 | 订单已关闭 | 重新创建订单 |
| 2001 | 余额不足 | 充值后重试 |
| 2002 | 支付超时 | 重新发起支付 |

## 其他说明

这是其他说明的内容。
"""

    print("\n" + "=" * 80)
    print("测试用例 2：错误码章节（应保持完整）")
    print("=" * 80)

    chunks2 = split_markdown_chunks(markdown2, chunk_size=512, chunk_overlap=100)

    print(f"\n总 chunk 数: {len(chunks2)}\n")

    error_chunks = [c for c in chunks2 if "错误码" in c["heading_path"]]

    if error_chunks:
        print(f"包含'错误码'的 chunk 数: {len(error_chunks)}")
        for i, chunk in enumerate(error_chunks, 1):
            tokens = estimate_token_size(chunk["text"])
            print(f"\n--- Chunk {i} ---")
            print(f"heading_path: {chunk['heading_path']}")
            print(f"tokens: {tokens}")

        if len(error_chunks) == 1:
            print("\n✅ 错误码章节保持完整（未被切分）")
        else:
            print(f"\n⚠️  错误码章节被切分成 {len(error_chunks)} 个 chunk")
    else:
        print("\n❌ 未找到错误码相关 chunk")

    # 测试用例 3：非原子章节（应正常切分）
    markdown3 = """
# 产品介绍

## 平台概述

这是一个非常长的产品介绍章节，包含大量的文字描述。我们的平台提供了丰富的功能，包括订单管理、支付管理、用户管理等。

平台采用微服务架构，保证了系统的高可用性和可扩展性。我们使用了最新的技术栈，包括 Kubernetes、Docker、Redis、MySQL 等。

系统支持高并发访问，能够处理每秒数万次请求。我们有专业的运维团队 7x24 小时监控系统运行状态。

""" + "\n\n".join([f"这是第 {i} 段内容，用于测试非原子章节的正常切分逻辑。" * 10 for i in range(20)])

    print("\n" + "=" * 80)
    print("测试用例 3：非原子章节（应正常切分）")
    print("=" * 80)

    chunks3 = split_markdown_chunks(markdown3, chunk_size=512, chunk_overlap=100)

    print(f"\n总 chunk 数: {len(chunks3)}")

    if len(chunks3) > 1:
        print("\n✅ 非原子章节正常切分")
        for i, chunk in enumerate(chunks3[:3], 1):
            tokens = estimate_token_size(chunk["text"])
            print(f"\nChunk {i}: {tokens} tokens")
    else:
        print("\n⚠️  预期应该切分成多个 chunk")


if __name__ == "__main__":
    test_atomic_section_chunking()

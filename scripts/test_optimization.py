#!/usr/bin/env python3
"""验证优化效果的测试脚本"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config.retrieval import QueryConfig
from rag.retrieval.query_service import build_query_service


def test_seed_selection_optimization():
    """测试 seed 选择优化"""
    print("=" * 80)
    print("测试 1：验证 seed 选择优化")
    print("=" * 80)

    cfg = QueryConfig.from_env()
    print(f"\n当前配置：")
    print(f"  seed_score_ratio: {cfg.dependency.seed_score_ratio}")
    print(f"  max_seeds: {cfg.dependency.max_seeds}")
    print(f"  seed_capability_weight: {cfg.dependency.seed_capability_weight}")
    print(f"  total_limit: {cfg.dependency.total_limit}")
    print(f"  final_top_n: {cfg.ranking.final_top_n}")
    print(f"  max_forced_dependencies: {cfg.dependency.max_forced_dependencies}")

    # 构建查询服务
    service = build_query_service(cfg)

    # 测试查询
    test_queries = [
        "生成订单接口如何接入加密？",
        "如何初始化 SDK？",
        "如何排查支付失败的问题？",
    ]

    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"查询: {query}")
        print(f"{'=' * 80}")

        try:
            result = service.query(query)

            # 分析结果
            print(f"\n召回统计:")
            print(f"  原始查询: {result.retrieval_result.query}")
            print(f"  改写查询数: {len(result.retrieval_result.rewritten_queries)}")
            print(f"  候选数: {len(result.retrieval_result.candidates)}")
            print(f"  最终上下文数: {len(result.retrieval_result.top_candidates)}")

            # 显示最终上下文的前几个
            print(f"\n最终上下文（前5个）:")
            for i, candidate in enumerate(result.retrieval_result.top_candidates[:5], 1):
                heading = candidate.metadata.get("heading_path", "")
                summary = candidate.metadata.get("summary", "")[:50]
                is_dep = candidate.is_dependency_evidence
                dep_mark = " [依赖证据]" if is_dep else ""
                print(f"  {i}. {heading}{dep_mark}")
                print(f"     摘要: {summary}...")
                print(f"     final_score: {candidate.final_score:.3f}")
                if hasattr(candidate, 'dependency_seed_score'):
                    print(f"     dependency_seed_score: {candidate.dependency_seed_score:.3f}")

            # 检查是否有强制依赖证据
            forced_deps = [c for c in result.retrieval_result.top_candidates if c.is_dependency_evidence]
            if forced_deps:
                print(f"\n✅ 强制依赖证据数量: {len(forced_deps)}")
                for dep in forced_deps:
                    heading = dep.metadata.get("heading_path", "")
                    print(f"   - {heading}")
            else:
                print(f"\n⚠️  未发现强制依赖证据")

            # 检查干扰文档
            generic_keywords = ["介绍", "概述", "简介", "背景", "产品"]
            interference_docs = [
                c for c in result.retrieval_result.top_candidates
                if any(kw in c.metadata.get("heading_path", "") for kw in generic_keywords)
            ]
            if interference_docs:
                print(f"\n⚠️  发现疑似干扰文档: {len(interference_docs)}")
                for doc in interference_docs:
                    print(f"   - {doc.metadata.get('heading_path', '')}")
            else:
                print(f"\n✅ 无干扰文档进入最终上下文")

            print(f"\n最终答案:")
            print(f"{result.answer}")

        except Exception as e:
            print(f"\n❌ 查询失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_seed_selection_optimization()

from __future__ import annotations

import re


TECH_TOKEN_PATTERN = re.compile(
    r"/[A-Za-z0-9_./-]+"
    r"|[A-Za-z]+[A-Za-z0-9_.-]*_[A-Za-z0-9_.-]+"
    r"|[A-Za-z]+[A-Za-z0-9_.-]*-[A-Za-z0-9_.-]+"
    r"|[A-Za-z]+(?:\.[A-Za-z0-9_-]+)+"
    r"|(?:[A-Za-z]+(?:\s+\d+(?:\.\d+)*)?)"
)
TOKEN_PLACEHOLDER_TEMPLATE = "__TECH_TOKEN_{index}__"
TOKEN_PLACEHOLDER_PATTERN = re.compile(r"__TECH_TOKEN_\d+__")
CHINESE_COMPOUND_WORDS = (
    "向量召回",
    "谷歌订阅",
    "订阅支付",
    "查询改写",
)
ALLOWED_SHORT_TOKENS = {"id", "md", "api", "ui", "llm"}


def tokenize_technical_text(text: str) -> list[str]:
    if not text:
        return []

    replaced_text, placeholders = _extract_technical_tokens(text)
    base_tokens = _tokenize_base_text(replaced_text)
    restored_tokens = _restore_placeholders(base_tokens, placeholders)
    merged_tokens = [_normalize_token(token) for token in restored_tokens]
    merged_tokens = [token for token in merged_tokens if _is_valid_token(token)]
    merged_tokens.extend(_collect_compound_words(text))
    return _deduplicate_preserving_order(merged_tokens)


def estimate_token_size(text: str) -> int:
    if not text:
        return 0
    replaced_text, placeholders = _extract_technical_tokens(text)
    base_tokens = _tokenize_base_text(replaced_text)
    restored_tokens = _restore_placeholders(base_tokens, placeholders)
    normalized_tokens = [_normalize_token(token) for token in restored_tokens]
    return sum(1 for token in normalized_tokens if _is_valid_token(token))


def _extract_technical_tokens(text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}
    replaced_parts: list[str] = []
    last_end = 0
    token_index = 0

    for match in TECH_TOKEN_PATTERN.finditer(text):
        token = _normalize_token(match.group(0))
        if not _is_meaningful_tech_token(token):
            continue
        placeholder = TOKEN_PLACEHOLDER_TEMPLATE.format(index=token_index)
        token_index += 1
        placeholders[placeholder] = token
        replaced_parts.append(text[last_end:match.start()])
        replaced_parts.append(f" {placeholder} ")
        last_end = match.end()

    replaced_parts.append(text[last_end:])
    return "".join(replaced_parts), placeholders


def _tokenize_base_text(text: str) -> list[str]:
    segments = re.split(f"({TOKEN_PLACEHOLDER_PATTERN.pattern})", text)
    tokens: list[str] = []
    for segment in segments:
        if not segment:
            continue
        if TOKEN_PLACEHOLDER_PATTERN.fullmatch(segment):
            tokens.append(segment)
            continue
        tokens.extend(_tokenize_plain_text(segment))
    return tokens


def _tokenize_plain_text(text: str) -> list[str]:
    try:
        import jieba
        return [token.strip() for token in jieba.lcut(text) if token.strip()]
    except ImportError:
        return re.findall(r"__TECH_TOKEN_\d+__|[A-Za-z0-9_./-]+|[\u4e00-\u9fff]+", text)


def _restore_placeholders(tokens: list[str], placeholders: dict[str, str]) -> list[str]:
    restored: list[str] = []
    for token in tokens:
        restored.append(placeholders.get(token, token))
    return restored


def _normalize_token(token: str) -> str:
    normalized = token.strip().strip("`\"'()[]{}<>:;,.!?，。；：、")
    if re.fullmatch(r"[A-Za-z0-9_./-]+", normalized):
        normalized = normalized.lower()
    return normalized


def _is_valid_token(token: str) -> bool:
    if not token:
        return False
    if re.fullmatch(r"[_./-]+", token):
        return False
    if re.fullmatch(r"[A-Za-z]", token) and token.lower() not in ALLOWED_SHORT_TOKENS:
        return False
    return True


def _is_meaningful_tech_token(token: str) -> bool:
    lowered = token.lower()
    if "/" in token or "_" in token or "-" in token or "." in token:
        return True
    if any(char.isdigit() for char in token) and any(char.isalpha() for char in token):
        return True
    return lowered in ALLOWED_SHORT_TOKENS


def _collect_compound_words(text: str) -> list[str]:
    compounds: list[str] = []
    compact_text = re.sub(r"\s+", "", text)
    for compound in CHINESE_COMPOUND_WORDS:
        if compound in compact_text:
            compounds.append(compound)
    return compounds


def _deduplicate_preserving_order(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered

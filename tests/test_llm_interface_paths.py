import os

from src.llm_interface import _resolve_dir


def test_resolve_dir_matches_existing_knowledge_base_folder():
    resolved = _resolve_dir("knowledge_base")
    assert os.path.basename(resolved) == "RAG knowledge base"
    assert os.path.exists(resolved)

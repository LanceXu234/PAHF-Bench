from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import threading
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger
from vita.environment.toolkit import ToolType, is_tool
from vita.memory.base import BaseMemory


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory.banks import DragonPlusEmbedding, FAISSMemoryBank, SQLiteMemoryBank  # noqa: E402
from utils.llm import LLMClient  # noqa: E402


_ENTRY_PATTERN = re.compile(
    r"^\[scope=(?P<scope>[^\]]+)\]"
    r"\[facet=(?P<facet>[^\]]+)\]"
    r"\[polarity=(?P<polarity>[^\]]+)\]"
    r"\[confidence=(?P<confidence>[^\]]+)\]\s*(?P<fact>.*)$"
)

_DELIVERY_KEYWORDS = (
    "外卖", "点个", "吃", "菜", "面", "奶茶", "咖啡", "火锅", "小吃", "送达", "配送"
)
_HOTEL_KEYWORDS = (
    "酒店", "住", "住宿", "民宿", "房间", "双床", "大床", "含早", "离地铁", "周边", "景区"
)
_TRAVEL_KEYWORDS = (
    "旅游", "旅行", "玩", "景点", "门票", "行程", "高铁", "机票", "出发", "返程"
)
_INSTORE_KEYWORDS = (
    "到店", "线下", "商场", "超市", "门店", "便利店", "药店", "实体店"
)
_SUBJECTIVE_KEYWORDS = (
    "适合我", "适合", "喜欢", "偏好", "帮我挑", "帮我选", "推荐", "想吃", "想住", "想去", "我会喜欢",
)

_GENERIC_EXTRACTION_PROMPT = """You are maintaining PAHF-style user memory for a life-service agent.

Your job is to extract a small set of future-useful personalized memory facts from a compact interaction digest.

Rules:
- Keep only user-specific preferences, dislikes, constraints, repeated choices, and recent preference changes.
- Ignore generic logistics unless they reveal a stable preference.
- Each fact must be atomic and concise.
- Scope must be one of: delivery_food, hotel, travel, instore, general.
- Polarity must be one of: prefer, avoid, constraint, interest, change.
- Facet should be a short slot name like spice_level, hotel_location, budget, room_type, merchant, item, schedule, brand, delivery_speed.
- Return strict JSON only.

JSON schema:
{
  "memories": [
    {
      "fact": "...",
      "scope": "delivery_food|hotel|travel|instore|general",
      "facet": "short_slot_name",
      "polarity": "prefer|avoid|constraint|interest|change",
      "confidence": 0.0,
      "evidence": "short evidence snippet"
    }
  ]
}
"""

_CLARIFICATION_PROMPT = """You are a PAHF-style assistant deciding whether one narrow clarification question is needed before acting.

Given the current user request and retrieved memory, decide whether the request still lacks key preference constraints.

Rules:
- Ask at most one narrow clarification question.
- Only ask when the current memory is clearly insufficient or conflicting for this request.
- If the memory is good enough, set should_ask to false.
- Return strict JSON only.

JSON schema:
{
  "should_ask": true,
  "trigger_type": "missing_state|conflict|none",
  "required_state_types": ["..."],
  "missing_state_types": ["..."],
  "conflicting_state_ids": ["..."],
  "trigger_reason": "...",
  "proposed_question": "..."
}
"""

_CLARIFICATION_FEEDBACK_PROMPT = """You are updating PAHF-style user memory from a clarification answer.

Convert the clarification answer into at most one durable personalized memory fact.
If the answer contains no useful preference information, return an empty memories list.
Return strict JSON only.

JSON schema:
{
  "memories": [
    {
      "fact": "...",
      "scope": "delivery_food|hotel|travel|instore|general",
      "facet": "short_slot_name",
      "polarity": "prefer|avoid|constraint|interest|change",
      "confidence": 0.0,
      "evidence": "short evidence snippet"
    }
  ]
}
"""


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "user"))
    cleaned = cleaned.strip("._-")
    return cleaned or "user"


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _stable_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _truncate(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _extract_json_block(text: str) -> Optional[Any]:
    raw = str(text or "").strip()
    if not raw:
        return None
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = raw.find(open_char)
        end = raw.rfind(close_char)
        if start != -1 and end != -1 and end > start:
            block = raw[start : end + 1]
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
    return None


def _env_flag(name: str, default: Optional[bool] = None) -> Optional[bool]:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


class PAHFMemory(BaseMemory):
    """PAHF-style preference memory bridged into VitaBench's native memory interface."""

    _shared_embedding_model: Optional[DragonPlusEmbedding] = None
    _shared_embedding_lock = threading.Lock()
    _extract_cache: Dict[str, List[Dict[str, Any]]] = {}
    _question_cache: Dict[str, Dict[str, Any]] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        language: str = None,
        top_k: int = 6,
        similarity_threshold: float = 0.45,
        user_id: Optional[str] = None,
        cache_root: Optional[str] = None,
        bank_type: str = "sql",
        enable_llm_extraction: bool = True,
        enable_llm_questions: bool = True,
        max_new_facts: int = 6,
        max_render_chars: int = 3200,
        clarification_similarity_threshold: float = 0.33,
        **kwargs,
    ):
        super().__init__(
            language=language,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            **kwargs,
        )
        env_bank_type = str(os.environ.get("PAHF_VITA_MEMORY_BACKEND", "")).strip().lower()
        if env_bank_type:
            bank_type = env_bank_type

        disable_extraction = _env_flag("PAHF_VITA_DISABLE_LLM_EXTRACTION")
        if disable_extraction is not None:
            enable_llm_extraction = not disable_extraction

        disable_questions = _env_flag("PAHF_VITA_DISABLE_LLM_QUESTIONS")
        if disable_questions is None:
            disable_questions = _env_flag("PAHF_VITA_DISABLE_LLM_QUESTION")
        if disable_questions is not None:
            enable_llm_questions = not disable_questions

        self.user_id = user_id or "user"
        self.bank_type = str(bank_type or "sql").lower()
        self.enable_llm_extraction = bool(enable_llm_extraction)
        self.enable_llm_questions = bool(enable_llm_questions)
        self.max_new_facts = int(max_new_facts)
        self.max_render_chars = int(max_render_chars)
        self.clarification_similarity_threshold = float(clarification_similarity_threshold)

        base_cache_root = (
            Path(cache_root)
            if cache_root
            else REPO_ROOT / "runs" / "vitabench_bridge_cache"
        )
        self.instance_id = uuid.uuid4().hex[:10]
        self.cache_root = base_cache_root / f"{_safe_name(self.user_id)}_{self.instance_id}"
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self._memory_bank = self._create_memory_bank()
        self._entries_by_id: Dict[int, Dict[str, Any]] = {}
        self._key_to_id: Dict[str, int] = {}
        self._text_to_ids: Dict[str, List[int]] = defaultdict(list)
        self._llm_client: Optional[LLMClient] = None
        self._llm_signature: Optional[Tuple[Any, ...]] = None
        self._last_query: str = ""
        self._last_package: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # BaseMemory interface
    # ------------------------------------------------------------------

    def read(self, query: str = None) -> str:
        normalized_query = str(query or "").strip()
        relevant_entries = self._retrieve_entries(normalized_query)
        package = self._build_inference_package(normalized_query, relevant_entries)
        self._last_query = normalized_query
        self._last_package = package
        return self._render_package(package)

    def update(
        self,
        new_interactions: list,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        **kwargs,
    ) -> str:
        if not new_interactions:
            return self.read(query=self._last_query)

        self._sync_llm_client(llm, llm_args)
        digest = self._build_interaction_digest(new_interactions)
        candidates = self._extract_candidates(digest, new_interactions)

        for candidate in candidates[: self.max_new_facts]:
            self._merge_or_add_candidate(candidate)

        logger.info(
            "PAHFMemory updated with {} candidate facts ({} stored memories).",
            len(candidates),
            len(self._entries_by_id),
        )
        return self.read(query=self._last_query)

    def reset(self):
        self._entries_by_id = {}
        self._key_to_id = {}
        self._text_to_ids = defaultdict(list)
        self._last_query = ""
        self._last_package = {}
        try:
            if hasattr(self._memory_bank, "close"):
                self._memory_bank.close()
        except Exception:
            pass
        if self.cache_root.exists():
            shutil.rmtree(self.cache_root, ignore_errors=True)

    # ------------------------------------------------------------------
    # Memory tools
    # ------------------------------------------------------------------

    @is_tool(ToolType.READ)
    def read_preference_memory(self) -> str:
        """Read the current PAHF preference memory."""
        return self.read(query=self._last_query)

    @is_tool(ToolType.READ)
    def query_preference_memory(self, query: str) -> str:
        """Query PAHF preference memory for task-relevant user preference evidence."""
        return self.read(query=query)

    @is_tool(ToolType.READ)
    def get_preference_clarification_hint(self, query: str) -> str:
        """Return whether PAHF believes a narrow clarification question is needed."""
        package = self.get_latest_inference_package(query=query, refresh=True)
        if not package.get("should_ask", False):
            return "Current PAHF memory looks sufficient; no clarification is strongly required."
        return (
            f"Clarification required ({package.get('trigger_type', 'missing_state')}).\n"
            f"Reason: {package.get('trigger_reason', '(none)')}\n"
            f"Suggested question: {package.get('proposed_question', '(none)')}"
        )

    # ------------------------------------------------------------------
    # Active clarification hooks used by PersonalizationAgent
    # ------------------------------------------------------------------

    def get_latest_inference_package(
        self,
        query: Optional[str] = None,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        normalized_query = str(query or "").strip()
        if refresh or not self._last_package or normalized_query != self._last_query:
            self.read(query=normalized_query)
        return json.loads(json.dumps(self._last_package, ensure_ascii=False, default=str))

    def apply_clarification_feedback(
        self,
        query: str,
        asked_question: str,
        user_feedback: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ) -> str:
        normalized_feedback = str(user_feedback or "").strip()
        if not normalized_feedback:
            return self.read(query=query)

        self._sync_llm_client(llm, llm_args)
        candidate = self._extract_candidate_from_feedback(query, asked_question, normalized_feedback)
        if candidate is not None:
            self._merge_or_add_candidate(candidate)
        return self.read(query=query)

    # ------------------------------------------------------------------
    # Bank lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def _get_shared_embedding_model(cls) -> DragonPlusEmbedding:
        with cls._shared_embedding_lock:
            if cls._shared_embedding_model is None:
                logger.info("Loading shared PAHF DragonPlus embedding model once for VitaBench bridge.")
                cls._shared_embedding_model = DragonPlusEmbedding()
            return cls._shared_embedding_model

    def _create_memory_bank(self):
        embedding_model = self._get_shared_embedding_model()
        person_id = self.user_id
        if self.bank_type == "faiss":
            return FAISSMemoryBank(
                embedding_model=embedding_model,
                person_id=person_id,
                persistence_path=str(self.cache_root / "memory_index"),
            )
        if self.bank_type != "sql":
            raise ValueError(f"Unsupported PAHF VitaBench bank_type: {self.bank_type}")
        return SQLiteMemoryBank(
            db_path=str(self.cache_root / "memory.db"),
            person_id=person_id,
            embedding_model=embedding_model,
        )

    def _sync_llm_client(self, llm: Optional[str], llm_args: Optional[dict]) -> None:
        llm_args = llm_args or {}
        model = llm or llm_args.get("model") or "gpt-4o-mini"
        api_key = llm_args.get("api_key")
        base_url = llm_args.get("base_url")
        timeout = llm_args.get("request_timeout") or llm_args.get("timeout") or 180
        max_retries = llm_args.get("max_retries") or 5
        signature = (model, api_key, base_url, timeout, max_retries)
        if self._llm_client is not None and signature == self._llm_signature:
            return
        if not api_key:
            self._llm_client = None
            self._llm_signature = None
            return
        self._llm_client = LLMClient(
            model=model,
            human_model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=float(timeout),
            max_retries=int(max_retries),
        )
        self._llm_signature = signature

    # ------------------------------------------------------------------
    # Extraction and merge
    # ------------------------------------------------------------------

    def _extract_candidates(
        self,
        digest: str,
        new_interactions: list,
    ) -> List[Dict[str, Any]]:
        if self.enable_llm_extraction and self._llm_client is not None:
            cache_key = f"extract::{self._llm_signature}::{_stable_hash(digest)}"
            with self._cache_lock:
                cached = self._extract_cache.get(cache_key)
            if cached is not None:
                return [dict(item) for item in cached]

            prompt = (
                f"{_GENERIC_EXTRACTION_PROMPT}\n\n"
                f"Interaction digest:\n{digest}\n"
            )
            try:
                response = self._llm_client.generate(prompt, temperature=0.0, max_tokens=1200)
                parsed = _extract_json_block(response)
                memories = parsed.get("memories", []) if isinstance(parsed, dict) else []
                candidates = [self._normalize_candidate(item) for item in memories]
                candidates = [item for item in candidates if item is not None]
                if candidates:
                    with self._cache_lock:
                        self._extract_cache[cache_key] = [dict(item) for item in candidates]
                    return candidates
            except Exception as exc:
                logger.warning("PAHFMemory LLM extraction failed; falling back to heuristics: {}", exc)

        return self._fallback_extract_candidates(new_interactions)

    def _extract_candidate_from_feedback(
        self,
        query: str,
        asked_question: str,
        user_feedback: str,
    ) -> Optional[Dict[str, Any]]:
        if self._llm_client is not None and self.enable_llm_extraction:
            prompt = (
                f"{_CLARIFICATION_FEEDBACK_PROMPT}\n\n"
                f"Current request: {query}\n"
                f"Clarification question: {asked_question}\n"
                f"User answer: {user_feedback}\n"
            )
            try:
                response = self._llm_client.generate(prompt, temperature=0.0, max_tokens=500)
                parsed = _extract_json_block(response)
                memories = parsed.get("memories", []) if isinstance(parsed, dict) else []
                for item in memories:
                    normalized = self._normalize_candidate(item)
                    if normalized is not None:
                        return normalized
            except Exception as exc:
                logger.warning("PAHFMemory clarification extraction failed; using fallback: {}", exc)

        scope = self._detect_scope(f"{query}\n{asked_question}\n{user_feedback}")
        return self._normalize_candidate(
            {
                "fact": f"User clarification for this request: {user_feedback}",
                "scope": scope,
                "facet": "clarification",
                "polarity": "constraint",
                "confidence": 0.7,
                "evidence": user_feedback,
            }
        )

    def _merge_or_add_candidate(self, candidate: Dict[str, Any]) -> None:
        entry_key = self._entry_key(candidate)
        existing_id = self._key_to_id.get(entry_key)

        if existing_id is None:
            similar_id = self._memory_bank.find_similar_memory(
                self._render_entry_text(candidate),
                threshold=self.similarity_threshold,
            )
            if similar_id is not None:
                similar_entry = self._entries_by_id.get(similar_id)
                if similar_entry is not None and self._entries_are_compatible(similar_entry, candidate):
                    existing_id = similar_id

        if existing_id is None:
            rendered = self._render_entry_text(candidate)
            self._memory_bank.add(rendered)
            memory_id = self._resolve_memory_id(rendered)
            if memory_id is None:
                return
            normalized = dict(candidate)
            normalized["memory_id"] = memory_id
            self._remember_entry(memory_id, rendered, normalized)
            return

        existing_entry = self._entries_by_id.get(existing_id)
        if existing_entry is None:
            rendered = self._render_entry_text(candidate)
            self._memory_bank.update_memory(existing_id, rendered)
            normalized = dict(candidate)
            normalized["memory_id"] = existing_id
            self._remember_entry(existing_id, rendered, normalized)
            return

        merged = self._merge_entries(existing_entry, candidate)
        rendered = self._render_entry_text(merged)
        self._memory_bank.update_memory(existing_id, rendered)
        merged["memory_id"] = existing_id
        self._remember_entry(existing_id, rendered, merged)

    def _merge_entries(self, existing: Dict[str, Any], new_item: Dict[str, Any]) -> Dict[str, Any]:
        existing_fact = str(existing.get("fact", "")).strip()
        new_fact = str(new_item.get("fact", "")).strip()
        merged_fact = new_fact

        if not new_fact:
            merged_fact = existing_fact
        elif new_item.get("polarity") == "change":
            merged_fact = new_fact
        elif existing_fact == new_fact:
            merged_fact = existing_fact
        elif new_fact in existing_fact:
            merged_fact = existing_fact
        elif existing_fact in new_fact:
            merged_fact = new_fact
        elif self._llm_client is not None:
            merge_prompt = (
                "Merge the following two user preference memory facts into one concise current-state fact. "
                "If the newer fact contradicts the older one, prefer the newer fact.\n\n"
                f"Old fact: {existing_fact}\n"
                f"New fact: {new_fact}\n"
                "Merged fact:"
            )
            try:
                merged_fact = self._llm_client.generate(
                    merge_prompt,
                    temperature=0.0,
                    max_tokens=200,
                ).strip() or new_fact
            except Exception:
                merged_fact = f"{existing_fact}; {new_fact}"
        else:
            merged_fact = f"{existing_fact}; {new_fact}"

        merged = dict(existing)
        merged.update(
            {
                "fact": _truncate(merged_fact, 360),
                "scope": new_item.get("scope") or existing.get("scope") or "general",
                "facet": new_item.get("facet") or existing.get("facet") or "general",
                "polarity": new_item.get("polarity") or existing.get("polarity") or "interest",
                "confidence": max(float(existing.get("confidence", 0.0)), float(new_item.get("confidence", 0.0))),
                "evidence": _truncate(
                    " | ".join(filter(None, [str(existing.get("evidence", "")).strip(), str(new_item.get("evidence", "")).strip()])),
                    280,
                ),
            }
        )
        return merged

    def _entries_are_compatible(self, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        if left.get("scope") != right.get("scope"):
            return False
        if left.get("facet") and right.get("facet") and left.get("facet") != right.get("facet"):
            return False
        return True

    def _resolve_memory_id(self, rendered_text: str) -> Optional[int]:
        text_to_match = str(rendered_text or "").strip()
        for memory_id, text in reversed(self._memory_bank.get_all_memories()):
            if str(text).strip() == text_to_match:
                return int(memory_id)
        return None

    def _remember_entry(self, memory_id: int, rendered_text: str, entry: Dict[str, Any]) -> None:
        old_entry = self._entries_by_id.get(memory_id)
        if old_entry is not None:
            old_key = self._entry_key(old_entry)
            if self._key_to_id.get(old_key) == memory_id:
                self._key_to_id.pop(old_key, None)
            old_ids = self._text_to_ids.get(old_entry.get("_rendered_text", ""), [])
            self._text_to_ids[old_entry.get("_rendered_text", "")] = [mid for mid in old_ids if mid != memory_id]

        normalized = dict(entry)
        normalized["memory_id"] = int(memory_id)
        normalized["_rendered_text"] = rendered_text
        self._entries_by_id[int(memory_id)] = normalized
        self._key_to_id[self._entry_key(normalized)] = int(memory_id)
        self._text_to_ids[rendered_text].append(int(memory_id))

    # ------------------------------------------------------------------
    # Candidate normalization / rendering
    # ------------------------------------------------------------------

    def _normalize_candidate(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        fact = _truncate(str(item.get("fact", "")).strip(), 360)
        if not fact:
            return None
        scope = str(item.get("scope", "general") or "general").strip().lower()
        if scope not in {"delivery_food", "hotel", "travel", "instore", "general"}:
            scope = "general"
        facet = re.sub(r"[^a-z0-9_]+", "_", str(item.get("facet", "general") or "general").strip().lower())
        facet = facet.strip("_") or "general"
        polarity = str(item.get("polarity", "interest") or "interest").strip().lower()
        if polarity not in {"prefer", "avoid", "constraint", "interest", "change"}:
            polarity = "interest"
        try:
            confidence = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        confidence = max(0.0, min(1.0, confidence))
        evidence = _truncate(str(item.get("evidence", "")).strip(), 220)
        return {
            "fact": fact,
            "scope": scope,
            "facet": facet,
            "polarity": polarity,
            "confidence": confidence,
            "evidence": evidence,
        }

    @staticmethod
    def _entry_key(entry: Dict[str, Any]) -> str:
        return f"{entry.get('scope', 'general')}|{entry.get('facet', 'general')}"

    def _render_entry_text(self, entry: Dict[str, Any]) -> str:
        return (
            f"[scope={entry.get('scope', 'general')}]"
            f"[facet={entry.get('facet', 'general')}]"
            f"[polarity={entry.get('polarity', 'interest')}]"
            f"[confidence={float(entry.get('confidence', 0.6)):.2f}] "
            f"{entry.get('fact', '').strip()}"
        )

    def _parse_entry_text(self, memory_id: int, text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        match = _ENTRY_PATTERN.match(raw)
        if not match:
            return {
                "memory_id": memory_id,
                "fact": raw,
                "scope": "general",
                "facet": "general",
                "polarity": "interest",
                "confidence": 0.5,
                "evidence": "",
                "_rendered_text": raw,
            }
        parsed = match.groupdict()
        return {
            "memory_id": memory_id,
            "fact": parsed["fact"].strip(),
            "scope": parsed["scope"].strip(),
            "facet": parsed["facet"].strip(),
            "polarity": parsed["polarity"].strip(),
            "confidence": float(parsed["confidence"]),
            "evidence": "",
            "_rendered_text": raw,
        }

    # ------------------------------------------------------------------
    # Retrieval and rendering
    # ------------------------------------------------------------------

    def _retrieve_entries(self, query: str) -> List[Dict[str, Any]]:
        if not self._entries_by_id:
            for memory_id, text in self._memory_bank.get_all_memories():
                self._remember_entry(int(memory_id), str(text), self._parse_entry_text(int(memory_id), str(text)))

        if not self._entries_by_id:
            return []

        if not query:
            items = list(self._entries_by_id.values())
            return items[-self.top_k :]

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for score, text in self._memory_bank.search(query, top_k=self.top_k):
            ids = self._text_to_ids.get(str(text).strip(), [])
            for memory_id in ids[:1]:
                entry = self._entries_by_id.get(memory_id)
                if entry is not None:
                    scored.append((float(score), dict(entry)))

        seen = set()
        unique_entries: List[Dict[str, Any]] = []
        for score, entry in scored:
            memory_id = entry.get("memory_id")
            if memory_id in seen:
                continue
            seen.add(memory_id)
            entry["score"] = score
            unique_entries.append(entry)
        return unique_entries

    def _build_inference_package(
        self,
        query: str,
        relevant_entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        state_summary = self._build_state_summary(relevant_entries)
        inventory = self._build_inventory(relevant_entries)
        cues = self._build_task_cues(query, relevant_entries)
        decision = self._build_clarification_decision(query, relevant_entries)
        trigger_summary = (
            f"Clarification required. {decision['trigger_reason']}\nSuggested question: {decision['proposed_question']}"
            if decision.get("should_ask", False)
            else "No clarification strongly required."
        )
        return {
            "current_state_summary": state_summary,
            "facet_inventory_summary": inventory,
            "task_aligned_cues": cues,
            "fine_grained_inventory": inventory,
            "trigger_summary": trigger_summary,
            **decision,
        }

    def _build_state_summary(self, relevant_entries: List[Dict[str, Any]]) -> str:
        if not relevant_entries:
            return "(none)"

        buckets: Dict[str, List[str]] = defaultdict(list)
        for entry in relevant_entries:
            scope = entry.get("scope", "general")
            buckets[scope].append(entry.get("fact", ""))

        lines = []
        for scope, facts in buckets.items():
            joined = "; ".join(_truncate(fact, 90) for fact in facts[:3])
            lines.append(f"- {scope}: {joined}")
        return "\n".join(lines)

    def _build_inventory(self, relevant_entries: List[Dict[str, Any]]) -> str:
        if not relevant_entries:
            return "(none)"
        lines = []
        for entry in relevant_entries:
            scope = entry.get("scope", "general")
            facet = entry.get("facet", "general")
            polarity = entry.get("polarity", "interest")
            fact = entry.get("fact", "")
            evidence = entry.get("evidence", "")
            line = f"- [{scope}/{facet}/{polarity}] {fact}"
            if evidence:
                line += f" | evidence: {evidence}"
            lines.append(line)
        return "\n".join(lines)

    def _build_task_cues(self, query: str, relevant_entries: List[Dict[str, Any]]) -> str:
        if not query:
            return "(none)"
        scope = self._detect_scope(query)
        lines = [f"- Current request scope: {scope}"]
        if relevant_entries:
            lines.append("- Use the fine-grained inventory below as the primary grounding for concrete choices.")
            top = relevant_entries[:3]
            for entry in top:
                lines.append(f"- Relevant cue: {entry.get('fact', '')}")
        else:
            lines.append("- Retrieved PAHF memory is sparse for this request.")
        return "\n".join(lines)

    def _build_clarification_decision(
        self,
        query: str,
        relevant_entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        scope = self._detect_scope(query)
        top_score = max((float(entry.get("score", 0.0)) for entry in relevant_entries), default=0.0)
        conflict_ids = self._detect_conflicts(relevant_entries)
        subjective = self._query_needs_preference_grounding(query)

        should_ask = False
        trigger_type = "none"
        trigger_reason = "Current PAHF memory appears sufficient for this request."
        missing_state_types: List[str] = []

        if conflict_ids:
            should_ask = True
            trigger_type = "conflict"
            trigger_reason = "Retrieved PAHF memory contains potentially conflicting preference evidence for this request."
        elif subjective and (not relevant_entries or top_score < self.clarification_similarity_threshold):
            should_ask = True
            trigger_type = "missing_state"
            trigger_reason = "Retrieved PAHF memory does not provide enough task-specific preference evidence for this request."
            missing_state_types = [scope]

        question = ""
        if should_ask:
            question = self._build_clarification_question(query, scope, relevant_entries)

        package = {
            "should_ask": should_ask,
            "trigger_type": trigger_type,
            "required_state_types": [scope] if scope else [],
            "missing_state_types": missing_state_types,
            "conflicting_state_ids": conflict_ids,
            "trigger_reason": trigger_reason,
            "proposed_question": question,
        }

        if should_ask and self.enable_llm_questions and self._llm_client is not None:
            cache_key = f"question::{self._llm_signature}::{_stable_hash({'query': query, 'entries': relevant_entries[:4], 'base': package})}"
            with self._cache_lock:
                cached = self._question_cache.get(cache_key)
            if cached is not None:
                return dict(cached)

            prompt = (
                f"{_CLARIFICATION_PROMPT}\n\n"
                f"Current request: {query}\n"
                f"Detected scope: {scope}\n"
                f"Retrieved memory:\n{self._build_inventory(relevant_entries[:4])}\n"
            )
            try:
                response = self._llm_client.generate(prompt, temperature=0.0, max_tokens=500)
                parsed = _extract_json_block(response)
                if isinstance(parsed, dict) and parsed.get("proposed_question"):
                    package.update(
                        {
                            "should_ask": bool(parsed.get("should_ask", True)),
                            "trigger_type": str(parsed.get("trigger_type", trigger_type) or trigger_type),
                            "required_state_types": list(parsed.get("required_state_types", package["required_state_types"]) or package["required_state_types"]),
                            "missing_state_types": list(parsed.get("missing_state_types", package["missing_state_types"]) or package["missing_state_types"]),
                            "conflicting_state_ids": list(parsed.get("conflicting_state_ids", package["conflicting_state_ids"]) or package["conflicting_state_ids"]),
                            "trigger_reason": str(parsed.get("trigger_reason", trigger_reason) or trigger_reason),
                            "proposed_question": str(parsed.get("proposed_question", question) or question),
                        }
                    )
                    with self._cache_lock:
                        self._question_cache[cache_key] = dict(package)
            except Exception as exc:
                logger.warning("PAHFMemory clarification LLM call failed; keeping heuristic question: {}", exc)

        return package

    def _build_clarification_question(
        self,
        query: str,
        scope: str,
        relevant_entries: List[Dict[str, Any]],
    ) -> str:
        if scope == "delivery_food":
            return "为了更符合你的口味，这次你最看重的是口味偏好、健康限制，还是送达时间？"
        if scope == "hotel":
            return "为了更符合你的住宿偏好，这次酒店你最看重的是位置、预算，还是房型和设施？"
        if scope == "travel":
            return "为了更符合你的出行偏好，这次你最看重的是时间安排、预算，还是路线和景点类型？"
        if scope == "instore":
            return "为了更符合你的偏好，这次你最看重的是品牌、价格，还是具体功能和规格？"
        return "为了更符合你的偏好，这次你最看重的约束是什么？"

    def _detect_conflicts(self, relevant_entries: List[Dict[str, Any]]) -> List[str]:
        buckets: Dict[Tuple[str, str], set] = defaultdict(set)
        ids: List[str] = []
        for entry in relevant_entries:
            key = (str(entry.get("scope", "")), str(entry.get("facet", "")))
            buckets[key].add(str(entry.get("polarity", "")))
        for entry in relevant_entries:
            key = (str(entry.get("scope", "")), str(entry.get("facet", "")))
            if len(buckets[key]) > 1:
                ids.append(str(entry.get("memory_id")))
        return ids

    def _render_package(self, package: Dict[str, Any]) -> str:
        sections = [
            "## Task-Aligned Preference Cues",
            package.get("task_aligned_cues", "(none)") or "(none)",
            "",
            "## Current User State",
            package.get("current_state_summary", "(none)") or "(none)",
            "",
            "## Fine-Grained Preference Inventory",
            package.get("fine_grained_inventory", "(none)") or "(none)",
            "",
            "## State Gaps / Clarification Hint",
            package.get("trigger_summary", "No clarification strongly required.") or "No clarification strongly required.",
        ]
        rendered = "\n".join(sections).strip()
        return _truncate(rendered, self.max_render_chars)

    # ------------------------------------------------------------------
    # Interaction digest and heuristic fallback
    # ------------------------------------------------------------------

    def _build_interaction_digest(self, new_interactions: list) -> str:
        search_terms: Counter[str] = Counter()
        item_terms: Counter[str] = Counter()
        tag_terms: Counter[str] = Counter()
        merchants: Counter[str] = Counter()
        dialogue_clues: List[str] = []
        behavior_lines: List[str] = []

        for interaction in new_interactions:
            if isinstance(interaction, dict) and "date" in interaction:
                date = interaction.get("date", "")
                for behavior in interaction.get("behavior", []):
                    if not isinstance(behavior, dict):
                        continue
                    behavior_type = behavior.get("behavior_type", "unknown")
                    content = behavior.get("content", {}) or {}
                    if behavior_type == "search":
                        keyword = str(content.get("keyword", "")).strip()
                        if keyword:
                            search_terms[keyword] += 1
                            behavior_lines.append(f"- [{date}] search: {keyword}")
                    elif behavior_type == "order":
                        merchant = str(content.get("merchant_name", "")).strip()
                        if merchant:
                            merchants[merchant] += 1
                        for tag in content.get("tags", []) or []:
                            if str(tag).strip():
                                tag_terms[str(tag).strip()] += 1
                        for item in content.get("items", []) or []:
                            if isinstance(item, dict):
                                name = str(item.get("product_name", "")).strip()
                                if name:
                                    item_terms[name] += 1
                        behavior_lines.append(
                            f"- [{date}] order: merchant={merchant or '?'} | tags={list(content.get('tags', []) or [])} | items={[item.get('product_name') for item in content.get('items', []) or [] if isinstance(item, dict)]}"
                        )
                for turn in interaction.get("dialogue", []) or []:
                    if not isinstance(turn, dict):
                        continue
                    if str(turn.get("role", "")).lower() != "user":
                        continue
                    utterance = str(turn.get("content", "")).strip()
                    if utterance:
                        dialogue_clues.append(utterance)
            elif isinstance(interaction, dict):
                itype = str(interaction.get("type", "")).strip()
                content = interaction.get("content", "")
                if itype == "search":
                    keyword = str(content.get("keyword", "") if isinstance(content, dict) else content).strip()
                    if keyword:
                        search_terms[keyword] += 1
                        behavior_lines.append(f"- search: {keyword}")
                else:
                    behavior_lines.append(f"- {itype}: {_truncate(_json_text(content), 140)}")

        lines = ["## Frequent cues"]
        if search_terms:
            lines.append("- Top search keywords: " + ", ".join(term for term, _ in search_terms.most_common(8)))
        if item_terms:
            lines.append("- Repeated ordered items: " + ", ".join(term for term, _ in item_terms.most_common(8)))
        if tag_terms:
            lines.append("- Repeated tags/preferences: " + ", ".join(term for term, _ in tag_terms.most_common(8)))
        if merchants:
            lines.append("- Repeated merchants: " + ", ".join(term for term, _ in merchants.most_common(6)))

        if behavior_lines:
            lines.extend(["", "## Behavior samples"])
            lines.extend(behavior_lines[:18])
        if dialogue_clues:
            lines.extend(["", "## User dialogue clues"])
            for utterance in dialogue_clues[-10:]:
                lines.append(f"- {utterance}")
        return "\n".join(lines)

    def _fallback_extract_candidates(self, new_interactions: list) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for interaction in new_interactions:
            if not isinstance(interaction, dict) or "date" not in interaction:
                continue
            date = interaction.get("date", "")
            for behavior in interaction.get("behavior", []) or []:
                if not isinstance(behavior, dict):
                    continue
                behavior_type = str(behavior.get("behavior_type", "")).strip()
                content = behavior.get("content", {}) or {}
                if behavior_type == "search":
                    keyword = str(content.get("keyword", "")).strip()
                    if keyword:
                        candidates.append(
                            {
                                "fact": f"The user has shown interest in '{keyword}'.",
                                "scope": self._detect_scope(keyword),
                                "facet": "search_interest",
                                "polarity": "interest",
                                "confidence": 0.55,
                                "evidence": keyword,
                            }
                        )
                elif behavior_type == "order":
                    tags = [str(tag).strip() for tag in content.get("tags", []) or [] if str(tag).strip()]
                    items = [str(item.get("product_name", "")).strip() for item in content.get("items", []) or [] if isinstance(item, dict) and str(item.get("product_name", "")).strip()]
                    merchant = str(content.get("merchant_name", "")).strip()
                    if tags:
                        candidates.append(
                            {
                                "fact": f"The user repeatedly orders food with these cues: {', '.join(tags[:4])}.",
                                "scope": "delivery_food",
                                "facet": "order_tags",
                                "polarity": "prefer",
                                "confidence": 0.62,
                                "evidence": ", ".join(tags[:4]),
                            }
                        )
                    if items:
                        candidates.append(
                            {
                                "fact": f"The user has ordered these items before: {', '.join(items[:4])}.",
                                "scope": "delivery_food",
                                "facet": "order_items",
                                "polarity": "interest",
                                "confidence": 0.58,
                                "evidence": ", ".join(items[:4]),
                            }
                        )
                    if merchant:
                        candidates.append(
                            {
                                "fact": f"The user has previously chosen merchant '{merchant}'.",
                                "scope": "delivery_food",
                                "facet": "merchant",
                                "polarity": "interest",
                                "confidence": 0.5,
                                "evidence": merchant,
                            }
                        )
            for turn in interaction.get("dialogue", []) or []:
                if not isinstance(turn, dict):
                    continue
                if str(turn.get("role", "")).lower() != "user":
                    continue
                utterance = str(turn.get("content", "")).strip()
                if not utterance:
                    continue
                lowered = utterance.lower()
                if "不辣" in utterance or "不要辣" in utterance or "吃不了辣" in utterance:
                    candidates.append(
                        {
                            "fact": "The user avoids spicy food.",
                            "scope": "delivery_food",
                            "facet": "spice_level",
                            "polarity": "avoid",
                            "confidence": 0.8,
                            "evidence": utterance,
                        }
                    )
                if "多加醋" in utterance or "多醋" in utterance:
                    candidates.append(
                        {
                            "fact": "The user likes extra vinegar in noodle-style food orders.",
                            "scope": "delivery_food",
                            "facet": "seasoning",
                            "polarity": "prefer",
                            "confidence": 0.74,
                            "evidence": utterance,
                        }
                    )
                if any(keyword in utterance for keyword in _HOTEL_KEYWORDS):
                    candidates.append(
                        {
                            "fact": f"User hotel/travel preference clue: {utterance}",
                            "scope": "hotel" if any(keyword in utterance for keyword in ('酒店', '住宿', '房间', '民宿')) else "travel",
                            "facet": "hotel_or_trip",
                            "polarity": "constraint",
                            "confidence": 0.56,
                            "evidence": utterance,
                        }
                    )
        normalized = [self._normalize_candidate(item) for item in candidates]
        dedup: Dict[str, Dict[str, Any]] = {}
        for item in normalized:
            if item is None:
                continue
            dedup[self._entry_key(item)] = item
        return list(dedup.values())

    # ------------------------------------------------------------------
    # Query understanding
    # ------------------------------------------------------------------

    def _detect_scope(self, text: str) -> str:
        normalized = str(text or "").lower()
        if any(keyword in normalized for keyword in _DELIVERY_KEYWORDS):
            return "delivery_food"
        if any(keyword in normalized for keyword in _HOTEL_KEYWORDS):
            return "hotel"
        if any(keyword in normalized for keyword in _TRAVEL_KEYWORDS):
            return "travel"
        if any(keyword in normalized for keyword in _INSTORE_KEYWORDS):
            return "instore"
        return "general"

    def _query_needs_preference_grounding(self, query: str) -> bool:
        normalized = str(query or "").lower()
        return any(token in normalized for token in _SUBJECTIVE_KEYWORDS) or self._detect_scope(normalized) != "general"

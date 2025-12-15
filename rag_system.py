import os
import logging
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------
# Prompt templates
# --------------------------------------------------------

QA_PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a collection of Jewish stories.

Use ONLY the context below to answer the question.
If the answer is not clearly supported by the context, say
"I don't know from these stories."

Context:
{context}

Question:
{question}
"""

EXPLANATION_PROMPT_TEMPLATE = """You are explaining why the following stories are relevant to a user's query.

User query:
{question}

Stories (with brief excerpts):
{context}

In 3–6 sentences, explain the main themes that connect these stories to the query.
Do NOT invent details that are not in the context.
"""

CHAT_PROMPT_TEMPLATE = """You are a helpful assistant discussing a single Jewish story with the user.

You must use ONLY the story text as your source of facts.
If the user asks about things not in the story, say
"I don't know from this story."

Story title: {title}
Book: {book}
Page: {page}
Source file: {source_file}

Story text:
{story_text}

Conversation so far:
{conversation}

User's latest message:
{latest_user_message}

In 2–5 sentences, reply helpfully and concretely while staying grounded in the story text.
"""


def _format_docs_for_context(docs: List[Any]) -> str:
    """Turn retrieved docs into a big text block for the LLM."""
    parts: List[str] = []
    prev_page_num = None  # Initialize the previous page number
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        book = meta.get("book_name", "Unknown book")
        title = meta.get("story_title", "Untitled story")
        page = meta.get("page", None)

        # Use the previous page number if page is not available
        if page is None:
            page = prev_page_num
        else:
            prev_page_num = page  # Update the previous page number if page is available

        header_bits = [f"{i}. {title} ({book})"]
        if page is not None:
            header_bits.append(f"page {page}")

        header = " - ".join(header_bits)
        parts.append(header)
        parts.append(doc.page_content)
        parts.append("")  # blank line between stories

    return "\n".join(parts).strip()


def _conversation_to_text(messages: List[Dict[str, str]]) -> str:
    """
    Convert a list of {"role": "user"/"assistant", "content": "..."} messages
    into a simple text transcript for the prompt.
    """
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        prefix = "User" if role == "user" else "Assistant"
        parts.append(f"{prefix}: {m.get('content', '')}")
    return "\n".join(parts).strip()


class StoryRAGSystem:
    """
    Wrapper around:
      - Chroma + sentence-transformer embeddings
      - Optional OpenAI LLM for explanation / QA / per-story chat

    Public methods:
      - search_stories(query, k)
      - get_story_with_context(query, k)
      - ask_question(question, k)
      - chat_about_story(story_dict, conversation_history)
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "story_collection",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        openai_api_key: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        self.debug = debug

        # ---- Embeddings (single, explicit load) ----
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # ---- Vector store ----
        self.vectordb = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectordb.as_retriever()

        # ---- LLM (optional) ----
        key = openai_api_key or os.getenv("OPENAI_API_KEY") or None
        self.llm_available = bool(openai_api_key)
        if key:
            logger.info("OpenAI key found; LLM features enabled.")
            self.llm = ChatOpenAI(
                api_key=key,
                model="gpt-4o-mini",
                temperature=0.2,
            )
        else:
            logger.info("No OpenAI key provided; running in retrieval-only mode.")
            self.llm = None

        # Prompts
        self.qa_prompt = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)
        self.explanation_prompt = ChatPromptTemplate.from_template(
            EXPLANATION_PROMPT_TEMPLATE
        )
        self.chat_prompt = ChatPromptTemplate.from_template(CHAT_PROMPT_TEMPLATE)

        # Chains (only created if LLM is available)
        if self.llm is not None:
            self._qa_chain = self.qa_prompt | self.llm | StrOutputParser()
            self._explanation_chain = (
                self.explanation_prompt | self.llm | StrOutputParser()
            )
            self._chat_chain = self.chat_prompt | self.llm | StrOutputParser()
        else:
            self._qa_chain = None
            self._explanation_chain = None
            self._chat_chain = None

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    @property
    def has_llm(self) -> bool:
        return self.llm is not None

    def update_openai_key(self, new_key: Optional[str]) -> None:
        """
        Allow the UI to inject a key later without recreating the whole system.
        """
        if new_key:
            logger.info("Updating OpenAI key and enabling LLM.")
            self.llm = ChatOpenAI(
                api_key=new_key,
                model="gpt-4o-mini",
                temperature=0.2,
            )
        else:
            logger.info("Disabling LLM (no key).")
            self.llm = None

        if self.llm is not None:
            self._qa_chain = self.qa_prompt | self.llm | StrOutputParser()
            self._explanation_chain = (
                self.explanation_prompt | self.llm | StrOutputParser()
            )
            self._chat_chain = self.chat_prompt | self.llm | StrOutputParser()
        else:
            self._qa_chain = None
            self._explanation_chain = None
            self._chat_chain = None

    def set_debug(self, debug: bool) -> None:
        """Let the UI flip debug mode on/off."""
        self.debug = debug

    # --------------------------------------------------------
    # Core public API
    # --------------------------------------------------------

    def search_stories(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Pure vector search.
        Returns a list of dicts with content, score, and metadata.
        """
        results = self.vectordb.similarity_search_with_score(query, k=k)

        formatted: List[Dict[str, Any]] = []
        # prev_known_page = None # For some reason, this is needed when it's run locally, but messes it up when deployed online
        for doc, score in results:
            meta = doc.metadata or {}
            # Chroma uses distance; smaller is better.
            # We'll convert to a crude "similarity percent" just for display.
            try:
                distance = float(score)
                similarity = max(0.0, ((2.0 - distance) / 2.0))
                similarity_pct = round(similarity * 100, 1)
                if self.debug:
                    print(f"Score: {score}, Distance: {distance}, Similarity: {similarity}, Similarity %: {similarity_pct}")
            except Exception:  # noqa: BLE001
                distance = float(score)
                similarity_pct = None

            # Get page number with fallback to previous known page
            page = meta.get("page")
            # Handle both None and "Unknown page" string (for backwards compatibility)
            if page is None or page == "Unknown page":
                page = prev_known_page
            else:
                # Update previous known page when we find a valid one
                prev_known_page = page

            formatted.append(
                {
                    "content": doc.page_content,
                    "score": distance,
                    "similarity_pct": similarity_pct,
                    "book_name": meta.get("book_name"),
                    "story_title": meta.get("story_title"),
                    "page": page,
                    "source_file": meta.get("source_file"),
                    "metadata": meta,
                }
            )

        return formatted

    def get_story_with_context(
        self, query: str, k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieval + (optional) natural-language explanation of why
        those stories matched the query.
        """
        stories = self.search_stories(query, k=k)

        # If no LLM, just return stories.
        if not self.has_llm or self._explanation_chain is None:
            return {"stories": stories, "explanation": None}

        # Build a compact context for explanation
        lines: List[str] = []
        for s in stories:
            title = s.get("story_title") or "Untitled story"
            book = s.get("book_name") or "Unknown book"
            snippet = (s.get("content") or "")[:400].replace("\n", " ")
            lines.append(f"{title} ({book}): {snippet}")

        context_text = "\n\n".join(lines)

        # DEBUG: print exact prompt sent to LLM
        if self.debug:
            full_prompt = EXPLANATION_PROMPT_TEMPLATE.format(
                context=context_text,
                question=query,
            )
            print("\n====== [DEBUG] EXPLANATION PROMPT ======\n")
            print(full_prompt)
            print("\n========================================\n")

        explanation = self._explanation_chain.invoke(
            {"context": context_text, "question": query}
        )

        return {"stories": stories, "explanation": explanation}

    def ask_question(
        self, question: str, k: int = 6
    ) -> Dict[str, Any]:
        """
        Classic RAG QA: retrieve docs, then have the LLM answer
        using only those docs as context.
        """
        docs = self.vectordb.similarity_search(question, k=k)

        if not self.has_llm or self._qa_chain is None:
            # Retrieval-only fallback
            return {
                "answer": None,
                "docs": docs,
            }

        context_text = _format_docs_for_context(docs)

        # DEBUG: print exact prompt sent to LLM
        if self.debug:
            full_prompt = QA_PROMPT_TEMPLATE.format(
                context=context_text,
                question=question,
            )
            print("\n========== [DEBUG] QA PROMPT ==========\n")
            print(full_prompt)
            print("\n=======================================\n")

        answer = self._qa_chain.invoke(
            {"context": context_text, "question": question}
        )

        return {
            "answer": answer,
            "docs": docs,
        }

    # --------------------------------------------------------
    # Per-story chat ("dive deeper")
    # --------------------------------------------------------

    def chat_about_story(
        self,
        story: Dict[str, Any],
        conversation: List[Dict[str, str]],
    ) -> str:
        """
        Chat about a *single* story, using only that story as context.

        `story` is one of the dicts returned by `search_stories`.
        `conversation` is a list of {"role": "user"/"assistant", "content": "..."}.

        Returns the assistant's reply as a plain string.
        """
        if not self.has_llm or self._chat_chain is None:
            raise RuntimeError("LLM not available; cannot use chat_about_story().")

        if not conversation:
            raise ValueError("conversation must contain at least one user message.")

        latest_user_msg = None
        for m in reversed(conversation):
            if m.get("role") == "user":
                latest_user_msg = m.get("content", "")
                break
        if latest_user_msg is None:
            latest_user_msg = conversation[-1].get("content", "")

        conv_text = _conversation_to_text(conversation)

        title = (
            story.get("story_title")
            or story.get("title")
            or "Untitled story"
        )
        book = (
            story.get("book_name")
            or story.get("book")
            or "Unknown book"
        )
        page = story.get("page", "?")
        source_file = story.get("source_file", "Unknown source")
        story_text = story.get("content", "")

        if self.debug:
            full_prompt = CHAT_PROMPT_TEMPLATE.format(
                title=title,
                book=book,
                page=page,
                source_file=source_file,
                story_text=story_text,
                conversation=conv_text,
                latest_user_message=latest_user_msg,
            )
            print("\n========== [DEBUG] CHAT PROMPT ==========\n")
            print(full_prompt)
            print("\n=========================================\n")

        reply = self._chat_chain.invoke(
            {
                "title": title,
                "book": book,
                "page": page,
                "source_file": source_file,
                "story_text": story_text,
                "conversation": conv_text,
                "latest_user_message": latest_user_msg,
            }
        )
        return reply

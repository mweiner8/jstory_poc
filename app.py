"""
Streamlit web application for story search using RAG.
Run with: streamlit run app.py
"""

import os
import json

import streamlit as st
from rag_system import StoryRAGSystem

# ------------------------------------------------------------
# PRE-LOAD: Initialize RAG system BEFORE any Streamlit UI
# This ensures the system loads during startup, not on first user click
# ------------------------------------------------------------

@st.cache_resource
def load_rag_system():
    """Load RAG system once and cache it across all sessions."""
    print("🔄 Initializing RAG system...", flush=True)
    try:
        system = StoryRAGSystem(
            persist_directory="./chroma_db",
            collection_name="story_collection",
            openai_api_key=None,  # Will be updated later from sidebar
            debug=False,
        )
        print("✅ RAG system initialized successfully!", flush=True)
        return system
    except Exception as ex:
        print(f"❌ Failed to initialize RAG system: {ex}", flush=True)
        import traceback
        traceback.print_exc()
        return None


# Load system at startup (outside any Streamlit widgets)
try:
    GLOBAL_RAG_SYSTEM = load_rag_system()
    if GLOBAL_RAG_SYSTEM is None:
        st.error("⚠️ Failed to load RAG system. Check logs for details.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Critical error loading RAG system: {e}")
    st.stop()

# ------------------------------------------------------------
# Page configuration + basic styling
# ------------------------------------------------------------

st.set_page_config(
    page_title="JStory RAG Explorer",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .story-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .story-title {
        font-size: 20px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .story-meta {
        color: #666;
        font-size: 14px;
        margin-bottom: 10px;
    }
    .story-content {
        color: #333;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

if "search_results" not in st.session_state:
    st.session_state["search_results"] = None

# For per-story chat "dive deeper"
if "active_story_idx" not in st.session_state:
    st.session_state["active_story_idx"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of {"role": ..., "content": ...}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def display_story(story: dict, index: int) -> None:
    """Display a single story result with robust key handling."""
    distance = story.get("score")
    if distance is not None:
        similarity = story.get(
            "similarity_pct",
            max(0, min(100, ((2.0 - distance) / 2.0) * 100)),
        )
    else:
        similarity = story.get("similarity_pct", 0.0)

    title = (
        story.get("title")
        or story.get("story_title")
        or "Untitled story"
    )

    book = (
        story.get("book")
        or story.get("book_name")
        or "Unknown book"
    )

    page = story.get("page")
    # Handle None, empty string, "Unknown page" string, or -1
    if page is None or page == "" or page == "Unknown page" or page == -1:
        page_display = "?"
    else:
        page_display = page

    content = story.get("content", "")

    st.markdown(
        f"""
        <div class="story-card">
            <div class="story-title">
                {index}. {title}
            </div>
            <div class="story-meta">
                📖 Book: {book} | 📄 Page: {page_display} | 🎯 Similarity: {similarity:.1f}%
            </div>
            <div class="story-content">
                {content[:500]}{"..." if len(content) > 500 else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("View full story text"):
        st.write(content)


def reset_chat_for_story(story_idx: int) -> None:
    """
    Select a new story for 'dive deeper' chat and reset chat history.
    """
    st.session_state["active_story_idx"] = story_idx
    st.session_state["chat_history"] = []


# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------

def main() -> None:
    st.title("📚 JStory RAG Explorer")
    st.markdown(
        "Type a theme, topic, or question, and I'll search your PDF-based story "
        "collection using embeddings + a vector database."
    )

    # Use the globally cached RAG system
    system: StoryRAGSystem = GLOBAL_RAG_SYSTEM
    if system is None:
        st.error("RAG system is not available.")
        st.stop()

    # ---------------- Sidebar: settings ----------------
    with st.sidebar:
        st.header("⚙️ Settings")

        default_key = os.getenv("OPENAI_API_KEY", "")

        openai_key_input = st.text_input(
            "OpenAI API Key (optional)",
            type="password",
            value=default_key,
            help=(
                "For AI explanations, QA, and per-story chat. "
                "Leave empty for retrieval-only search."
            ),
        )

        top_k = st.slider(
            "Number of results",
            min_value=1,
            max_value=20,
            value=3,
            help="How many stories to retrieve",
        )

        debug_mode = st.checkbox(
            "Debug mode: print prompts to console",
            value=False,
        )

        # Update system settings live
        system.update_openai_key(openai_key_input or None)
        system.set_debug(debug_mode)

        st.markdown("---")
        st.markdown("#### About")
        st.info(
            "This app uses RAG (Retrieval-Augmented Generation) "
            "to find stories matching your query from a collection "
            "of books stored in a Chroma vector database."
        )

        st.markdown("#### Tech Stack")
        st.markdown(
            """
            - **LangChain**: RAG orchestration  
            - **ChromaDB**: Vector database  
            - **HuggingFace**: Embeddings  
            - **OpenAI** (optional): Explanations, QA & per-story chat  
            - **Streamlit**: Web interface
            """
        )

    # ---------------- Main content ----------------
    st.markdown("### 🔍 Search for Stories")

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Search query or question",
            placeholder="e.g. 'honesty', 'keeping Shabbos at work', 'jealousy', ...",
            label_visibility="collapsed",
        )
    with col2:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    st.caption(
        "**Example queries:** honesty, jealousy, Shabbos at work, "
        "teshuvah, emunah, friendship, bitachon"
    )

    if not query.strip():
        if st.session_state["search_results"] is None:
            st.info("Enter a search query above to get started.")
        search_clicked = False

    if search_clicked and query.strip():
        with st.spinner("Searching stories..."):
            try:
                if system.has_llm:
                    raw = system.get_story_with_context(query, k=top_k)
                    results = {"query": query, **raw}
                else:
                    stories = system.search_stories(query, k=top_k)
                    results = {
                        "query": query,
                        "stories": stories,
                        "explanation": None,
                    }
                st.session_state["search_results"] = results
                # Reset per-story chat when doing a new search
                st.session_state["active_story_idx"] = None
                st.session_state["chat_history"] = []
            except Exception as e:
                st.error(f"Search error: {e}")

    # ---------------- Results display ----------------
    results = st.session_state["search_results"]
    if results:
        st.markdown("---")
        st.markdown(f"### Results for: *\"{results['query']}\"*")

        explanation = results.get("explanation")
        if explanation:
            st.subheader("Why these stories?")
            st.write(explanation)

        stories = results.get("stories") or []
        st.subheader(f"Top {len(stories)} matching stories")

        if not stories:
            st.warning("No stories were retrieved from the vector database.")
        else:
            for i, story in enumerate(stories, 1):
                # Display the story card
                display_story(story, i)

                # "Dive deeper" button for this story
                if system.has_llm:
                    if st.button(
                        "💬 Dive deeper with this story",
                        key=f"dive_{i}",
                    ):
                        reset_chat_for_story(i - 1)
                else:
                    st.caption("Connect an OpenAI key in the sidebar to enable chat.")

            # JSON export
            st.markdown("---")
            exportable_results = stories
            json_str = json.dumps(exportable_results, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 Download results as JSON",
                file_name="jstory_results.json",
                mime="application/json",
                data=json_str,
            )

        # ---------------- Per-story chat section ----------------
        active_idx = st.session_state["active_story_idx"]
        if active_idx is not None and 0 <= active_idx < len(stories):
            st.markdown("---")
            st.markdown("### 💬 Chatbot: Dive deeper into a specific story")

            if not system.has_llm:
                st.info("Connect an OpenAI key in the sidebar to enable the chatbot.")
            else:
                active_story = stories[active_idx]

                title = (
                    active_story.get("story_title")
                    or active_story.get("title")
                    or "Untitled story"
                )
                book = (
                    active_story.get("book_name")
                    or active_story.get("book")
                    or "Unknown book"
                )
                page = active_story.get("page", "?")

                st.markdown(
                    f"**Currently chatting about:** *{title}* "
                    f"from **{book}** (page {page})"
                )

                # Show chat history
                for msg in st.session_state["chat_history"]:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

                # Chat input
                user_msg = st.chat_input(
                    "Ask a question or share a thought about this story"
                )

                if user_msg:
                    # Append user message
                    st.session_state["chat_history"].append(
                        {"role": "user", "content": user_msg}
                    )

                    try:
                        with st.spinner("Thinking about this story..."):
                            reply = system.chat_about_story(
                                active_story,
                                st.session_state["chat_history"],
                            )
                    except Exception as e:
                        reply = f"Error during chat: {e}"

                    # Append assistant reply
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": reply}
                    )

                    # Immediately display last turn so user sees it without rerun confusion
                    with st.chat_message("user"):
                        st.write(user_msg)

                    with st.chat_message("assistant"):
                        st.write(reply)


if __name__ == "__main__":
    main()

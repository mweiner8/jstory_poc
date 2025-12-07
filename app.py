"""
Streamlit web application for story search using RAG.
Run with: streamlit run app.py
"""

import os
import json

import streamlit as st
from rag_system import StoryRAGSystem

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

if "rag_system" not in st.session_state:
    st.session_state["rag_system"] = None

if "search_results" not in st.session_state:
    st.session_state["search_results"] = None


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def display_story(story: dict, index: int) -> None:
    """Display a single story result with robust key handling."""
    distance = story.get("score")
    if distance is not None:
        similarity = story.get(
            "similarity_pct",
            max(0, min(100, (1.5 - distance) / 1.5 * 100)),
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
    page_display = page if page not in (None, "", -1) else "?"

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


# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------

def main() -> None:
    st.title("📚 JStory RAG Explorer")
    st.markdown(
        "Type a theme, topic, or question, and I'll search your PDF-based story "
        "collection using embeddings + a vector database."
    )

    # ---------------- Sidebar: settings ----------------
    with st.sidebar:
        st.header("⚙️ Settings")

        default_key = os.getenv("OPENAI_API_KEY", "")

        openai_key_input = st.text_input(
            "OpenAI API Key (optional)",
            type="password",
            value=default_key,
            help="For AI explanations and QA. Leave empty for retrieval-only search.",
        )

        top_k = st.slider(
            "Number of results",
            min_value=1,
            max_value=20,
            value=5,
            help="How many stories to retrieve",
        )

        debug_mode = st.checkbox(
            "Debug mode: print prompts to console",
            value=False,
        )

        init_clicked = st.button("🔄 Initialize / Reload System", type="primary")

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
            - **OpenAI** (optional): Explanations & QA  
            - **Streamlit**: Web interface
            """
        )

    # ---------------- RAG system init / update ----------------
    if init_clicked or st.session_state["rag_system"] is None:
        try:
            with st.spinner("Loading vector database and embeddings..."):
                st.session_state["rag_system"] = StoryRAGSystem(
                    persist_directory="./chroma_db",
                    collection_name="story_collection",
                    openai_api_key=openai_key_input or None,
                    debug=debug_mode,
                )
        except Exception as e:
            st.session_state["rag_system"] = None
            st.error(f"Error initializing system: {e}")
            return
    else:
        # Live update key + debug on existing system
        system = st.session_state["rag_system"]
        system.update_openai_key(openai_key_input or None)
        system.set_debug(debug_mode)

    system: StoryRAGSystem = st.session_state["rag_system"]

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
                display_story(story, i)

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


if __name__ == "__main__":
    main()

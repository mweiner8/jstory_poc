"""
Streamlit web application for story search using RAG.
Run with: streamlit run app.py
"""

import streamlit as st
from rag_system import StoryRAGSystem
import os

# Page configuration
st.set_page_config(
    page_title="Story Finder",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
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
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None


def initialize_rag_system(api_key=None):
    """Initialize the RAG system."""
    try:
        with st.spinner("Loading vector database and embeddings..."):
            rag_system = StoryRAGSystem(
                chroma_db_dir="./chroma_db",
                collection_name="story_collection",
                openai_api_key=api_key
            )
        st.session_state.rag_system = rag_system
        return True
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        return False


def display_story(story, index):
    """Display a single story result."""
    # ChromaDB returns distance (lower is better)
    # Convert to similarity: typical distances are 0.3-1.5
    distance = story['score']
    similarity = max(0, min(100, (1.5 - distance) / 1.5 * 100))

    st.markdown(f"""
        <div class="story-card">
            <div class="story-title">
                {index}. {story['title']}
            </div>
            <div class="story-meta">
                📖 Book: {story['book']} | 📄 Page: {story['page']} | 🎯 Similarity: {similarity:.1f}%
            </div>
            <div class="story-content">
                {story['content'][:500]}{"..." if len(story['content']) > 500 else ""}
            </div>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("View full content"):
        st.write(story['content'])


def main():
    """Main application."""

    # Header
    st.title("📚 Story Finder")
    st.markdown("### Find the perfect story using AI-powered search")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # OpenAI API Key (optional)
        st.markdown("#### OpenAI API Key (Optional)")
        st.caption("For enhanced explanations. Leave empty for basic search.")
        api_key = st.text_input(
            "API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Optional: Add OpenAI API key for AI-generated explanations"
        )

        # Number of results
        num_results = st.slider(
            "Number of results",
            min_value=1,
            max_value=10,
            value=3,
            help="How many stories to retrieve"
        )

        # Initialize button
        if st.button("🔄 Initialize System", type="primary"):
            initialize_rag_system(api_key if api_key else None)

        st.markdown("---")

        # Info
        st.markdown("#### About")
        st.info(
            "This app uses RAG (Retrieval-Augmented Generation) "
            "to find stories matching your query from a collection "
            "of books stored in a vector database."
        )

        st.markdown("#### Tech Stack")
        st.markdown("""
        - **LangChain**: RAG orchestration
        - **ChromaDB**: Vector database
        - **HuggingFace**: Embeddings
        - **Streamlit**: Web interface
        """)

    # Main content
    if st.session_state.rag_system is None:
        st.info("👈 Click 'Initialize System' in the sidebar to get started")

        st.markdown("""
        ### How it works:
        1. **Initialize** the system (loads the vector database)
        2. **Enter** a query describing the story you're looking for
        3. **Get** the top matching stories from the collection

        ### Example queries:
        - "adventure story with a brave hero"
        - "fairy tale about kindness"
        - "mystery involving a detective"
        - "story about friendship and loyalty"
        """)
    else:
        # Search interface
        st.markdown("### 🔍 Search for Stories")

        col1, col2 = st.columns([4, 1])

        with col1:
            query = st.text_input(
                "What kind of story are you looking for?",
                placeholder="e.g., adventure story with dragons...",
                label_visibility="collapsed"
            )

        with col2:
            search_button = st.button("Search", type="primary", use_container_width=True)

        # Example queries
        st.caption("**Example queries:** adventure story, fairy tale, mystery, friendship")

        # Perform search
        if search_button and query:
            with st.spinner("Searching through stories..."):
                try:
                    # Get results with context if OpenAI is available
                    if st.session_state.rag_system.llm_available:
                        results = st.session_state.rag_system.get_story_with_context(query, k=num_results)
                        st.session_state.search_results = results
                    else:
                        stories = st.session_state.rag_system.search_stories(query, k=num_results)
                        st.session_state.search_results = {
                            'query': query,
                            'stories': stories,
                            'explanation': None
                        }
                except Exception as e:
                    st.error(f"Search error: {e}")

        # Display results
        if st.session_state.search_results:
            results = st.session_state.search_results

            st.markdown("---")
            st.markdown(f"### Results for: *\"{results['query']}\"*")

            # Show AI explanation if available
            if results.get('explanation'):
                st.info(f"💡 {results['explanation']}")

            st.markdown(f"Found **{len(results['stories'])}** matching stories:")

            # Display each story
            for i, story in enumerate(results['stories'], 1):
                display_story(story, i)

            # Download results option
            st.markdown("---")
            if st.button("📥 Export Results"):
                import json
                results_json = json.dumps(results, indent=2)
                st.download_button(
                    label="Download as JSON",
                    data=results_json,
                    file_name="story_search_results.json",
                    mime="application/json"
                )

        # Show sample search if no results yet
        elif query == "":
            st.info("💡 Enter a search query above to find matching stories!")


if __name__ == "__main__":
    main()
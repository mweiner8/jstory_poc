"""
RAG query system for story retrieval.
This module handles querying the vector database and generating responses.
Uses LCEL (LangChain Expression Language) - works with all LangChain versions.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from typing import List, Dict
import os

class StoryRAGSystem:
    """RAG system for story retrieval and recommendation."""

    def __init__(
        self,
        chroma_db_dir: str = "./chroma_db",
        collection_name: str = "story_collection",
        openai_api_key: str = None
    ):
        """Initialize the RAG system."""

        self.chroma_db_dir = chroma_db_dir
        self.collection_name = collection_name

        # Set OpenAI API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        # Load embeddings model (same as used for creating DB)
        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.embeddings = HuggingFaceEmbeddings(
            # model_name="sentence-transformers/all-mpnet-base-v2",  # More accurate
            # OR
            model_name="BAAI/bge-small-en-v1.5",  # Specifically good for retrieval
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Load vector database
        print("Loading vector database...")
        self.vectordb = Chroma(
            persist_directory=chroma_db_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )

        # Initialize LLM (optional - for enhanced responses)
        try:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7
            )
            self.llm_available = True
        except Exception as e:
            print(f"LLM not available: {e}")
            self.llm = None
            self.llm_available = False

        print("RAG system initialized successfully!")

    def search_stories(self, my_query: str, k: int = 3) -> List[Dict]:
        """Search for stories matching the query."""

        # Get more results than needed to account for duplicates
        the_results = self.vectordb.similarity_search_with_score(my_query, k=k * 2)

        stories = []
        seen_content = set()

        for doc, score in the_results:
            # Skip duplicates (check first 100 chars)
            content_hash = doc.page_content[:100]
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)

            the_story = {
                'content': doc.page_content,
                'score': float(score),
                'metadata': doc.metadata,
                'source': doc.metadata.get('source_file', 'Unknown'),
                'book': doc.metadata.get('book_name', 'Unknown'),
                'title': doc.metadata.get('story_title', 'Untitled'),
                'page': doc.metadata.get('page', 'Unknown')
            }
            stories.append(the_story)

            # Stop when we have enough unique results
            if len(stories) >= k:
                break

        return stories

    def get_story_with_context(self, my_query: str, k: int = 3) -> Dict:
        """
        Get stories with AI-generated context and explanation.
        Uses LCEL (LangChain Expression Language) for the chain.

        Args:
            my_query: User's search query
            k: Number of results to return

        Returns:
            Dictionary with stories and AI explanation
        """

        # Get matching stories
        stories = self.search_stories(my_query, k=k)

        if not self.llm_available:
            return {
                'query': my_query,
                'stories': stories,
                'explanation': None
            }

        try:
            # Create retriever
            retriever = self.vectordb.as_retriever(search_kwargs={"k": k})

            # Create prompt
            system_template = """You are a helpful librarian assistant. 
Based on the user's query and the stories found, provide a brief explanation 
of why these stories match and what the user can expect.

Context:
{context}

Provide a friendly 2-3 sentence explanation of why these stories match the query."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_template),
                ("human", "{question}")
            ])

            # Helper function to format documents
            def format_docs(docs):
                return "\n\n".join(doc.page_content[:200] for doc in docs)

            # Create LCEL chain
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Get explanation
            explanation = rag_chain.invoke(my_query)

            return {
                'query': my_query,
                'stories': stories,
                'explanation': explanation
            }

        except Exception as e:
            print(f"Error generating explanation: {e}")
            return {
                'query': my_query,
                'stories': stories,
                'explanation': None
            }

    def get_retriever(self, k: int = 3):
        """Get a LangChain retriever for use in chains."""
        return self.vectordb.as_retriever(search_kwargs={"k": k})

    def ask_question(self, question: str, k: int = 3) -> str:
        """
        Ask a question and get an answer based on the story collection.
        Uses simple LCEL chain - works with all LangChain versions.

        Args:
            question: Question to ask
            k: Number of documents to retrieve

        Returns:
            Answer string
        """

        if not self.llm_available:
            raise ValueError("LLM not available. Set OPENAI_API_KEY to use this feature.")

        # Create retriever
        retriever = self.get_retriever(k=k)

        # Create prompt
        template = """Use the following pieces of context to answer the question.
If you don't know the answer, just say you don't know.
Use three sentences maximum and keep the answer concise.

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        # Helper function to format documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Create LCEL chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Get answer
        answer = rag_chain.invoke(question)
        return answer

# Example usage
if __name__ == "__main__":
    # Initialize system
    rag_system = StoryRAGSystem()

    # Test query
    query = "adventure story with a hero"
    print(f"\nSearching for: '{query}'")

    # Get results
    results = rag_system.search_stories(query, k=3)

    print(f"\nFound {len(results)} matching stories:\n")

    for i, story in enumerate(results, 1):
        print(f"{i}. {story['title']}")
        print(f"   Book: {story['book']}")
        print(f"   Relevance Score: {story['score']:.4f}")
        print(f"   Preview: {story['content'][:200]}...")
        print()
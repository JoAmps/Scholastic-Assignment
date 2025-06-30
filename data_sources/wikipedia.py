from langchain_community.retrievers import WikipediaRetriever
from langchain.schema import Document
from typing import List

# Initialize the retriever once
retriever = WikipediaRetriever(top_k_results=1)


def fetch_wikipedia_information(query: str) -> str:
    """
    Retrieves a summary from Wikipedia using LangChain's WikipediaRetriever.

    Args:
        query (str): The topic to search for on Wikipedia.

    Returns:
        str: A formatted string containing the retrieved Wikipedia content.
    """
    try:
        docs: List[Document] = retriever.invoke(query)
        return (
            "\n\n".join(doc.page_content for doc in docs)
            if docs
            else "No information found."
        )
    except Exception as e:
        print(f"Error retrieving Wikipedia data: {e}")
        return "Error retrieving data from Wikipedia."

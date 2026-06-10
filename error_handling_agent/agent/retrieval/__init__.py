"""RAG layer for the error-handling agent.

Loads the SAP CDS reference PDFs + crawled help-portal pages into a persistent
Chroma vector store keyed by local MiniLM embeddings, and exposes a LangGraph
node that pulls the most relevant excerpts for the (CDS, error) pair the user
pasted so the fix prompt can be grounded in real documentation.
"""

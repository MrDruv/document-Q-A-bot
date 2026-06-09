# prompts.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Key design decisions for voice:
# 1. Ask for SHORT answers — TTS sounds bad with walls of text
# 2. No markdown — voice can't speak bullet points
# 3. Cite sources inline, not in footnotes

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful voice assistant. Answer using ONLY the context below.

Rules:
- Keep answers under 3 sentences for voice clarity
- Never use bullet points, headers, or markdown
- If the answer isn't in the context, say: "I don't have that information."
- Cite your source naturally: "According to [source]..."

Context:
{context}
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Query rewriter — makes retrieval much better
REWRITER_PROMPT = ChatPromptTemplate.from_template("""
You are a query optimizer. Rewrite the user's voice query to be better 
suited for semantic document search. Remove filler words, expand 
abbreviations, and make it specific.

Original query: {question}
Rewritten query:""")
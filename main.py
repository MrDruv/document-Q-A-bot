# main.py
import asyncio
from documents import load_and_split_docs   # your doc loading logic
from retriever import RAGRetriever
from memory import AgentMemory
from graph import build_graph
from voice import VoiceInput, VoiceOutput
from dotenv import load_dotenv
load_dotenv()

async def main():
    print("Loading documents...")
    docs = load_and_split_docs("./data/")    # your PDFs, CSVs, etc.
    
    print("Building retriever...")
    retriever = RAGRetriever(docs)
    
    memory = AgentMemory(k=5)
    graph = build_graph(retriever, memory)
    
    voice_in = VoiceInput()
    voice_out = VoiceOutput()
    
    print("System ready. Press Ctrl+C to exit.\n")
    
    while True:
        # Record and transcribe
        audio = voice_in.record_until_silence()
        question = voice_in.transcribe(audio)
        
        if not question:
            continue
        
        print(f"You: {question}")
        
        # Run graph
        state = {
            "question": question,
            "retry_count": 0,
            "answer_tokens": [],
            "documents": [],
            "hallucination_score": 1.0,
        }
        
        result = graph.invoke(state)
        print(f"Agent: {result['answer']}")
        
        # Stream to speaker sentence by sentence
        voice_out.speak(result["answer"])

if __name__ == "__main__":
    asyncio.run(main())
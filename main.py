# main.py
import asyncio
from voice import VoiceInput, VoiceOutput
from state import make_initial_state
from pipeline import init_pipeline
from dotenv import load_dotenv
load_dotenv()

async def main():
    print("Initializing pipeline...")
    _retriever, _memory, graph = init_pipeline("./data/")
    
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
        
        result = graph.invoke(make_initial_state(question))
        print(f"Agent: {result['answer']}")
        
        # Stream to speaker sentence by sentence
        voice_out.speak(result["answer"])

if __name__ == "__main__":
    asyncio.run(main())
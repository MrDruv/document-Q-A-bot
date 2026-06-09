# memory.py
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage

class AgentMemory:
    """
    Window memory keeps the last N turns.
    Design decision: K=5 for voice agents — enough context, 
    small enough to not blow the context window.
    """
    def __init__(self, k: int = 5):
        self.memory = ConversationBufferWindowMemory(
            k=k,
            return_messages=True,
            memory_key="chat_history"
        )
    
    def get_history(self) -> list:
        return self.memory.load_memory_variables({})["chat_history"]
    
    def save_turn(self, question: str, answer: str):
        self.memory.save_context(
            {"input": question},
            {"output": answer}
        )
    
    def clear(self):
        self.memory.clear()
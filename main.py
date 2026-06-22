# main.py
import asyncio
import logging
import sys
from voice import VoiceInput, VoiceOutput
from state import make_initial_state
from pipeline import init_pipeline
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

async def main():
    try:
        logger.info("Initializing pipeline...")
        _retriever, _memory, graph = init_pipeline("./data/")
    except Exception as exc:
        logger.critical("Failed to initialize pipeline: %s", exc)
        sys.exit(1)

    try:
        voice_in = VoiceInput()
        voice_out = VoiceOutput()
    except RuntimeError as exc:
        logger.critical("Failed to initialize voice I/O: %s", exc)
        sys.exit(1)

    logger.info("System ready. Press Ctrl+C to exit.")

    while True:
        try:
            audio = voice_in.record_until_silence()
            question = voice_in.transcribe(audio)
        except RuntimeError as exc:
            logger.error("Voice input error: %s", exc)
            continue

        if not question:
            continue

        logger.info("You: %s", question)

        try:
            result = graph.invoke(make_initial_state(question))
        except Exception as exc:
            logger.error("Graph invocation failed: %s", exc)
            voice_out.speak("Sorry, I encountered an error processing your question.")
            continue

        answer = result.get("answer", "Sorry, I could not generate an answer.")
        logger.info("Agent: %s", answer)

        try:
            voice_out.speak(answer)
        except RuntimeError as exc:
            logger.error("Voice output error: %s", exc)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down.")

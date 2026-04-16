import os
import uuid

from line.events import (
    AgentSendText, AgentSendCustom, UserCustomSent, UserTurnEnded, UserTextSent, CallStarted, CallEnded, UserTurnStarted
)
from loguru import logger

from line.llm_agent import LlmAgent, LlmConfig, web_search
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

class TranscriptWrapper:
    def __init__(self, inner_agent):
        self.inner = inner_agent

    async def _process_inner(self, env, event):
        """Standardize internal proxy output handling and intercept AI responses."""
        if isinstance(event, UserCustomSent):
            # Handle custom events here; do not pass them to the inner LlmAgent.
            return
        started = False
        message_id = None
        transcript = ''
        async for output in self.inner.process(env, event):
            if isinstance(output, AgentSendText):
                if not started:
                    started = True
                    message_id = str(uuid.uuid4())
                    yield AgentSendCustom(metadata={
                        "id": message_id,
                        "role": "assistant",
                        "type": "start"
                    })
                yield output
                transcript += output.text
                yield AgentSendCustom(metadata={
                    "id": message_id,
                    "role": "assistant",
                    "type": "chunk",
                    "transcript": output.text
                })
            else:
                yield output
        if started:
            yield AgentSendCustom(metadata={
                "id": message_id,
                "role": "assistant",
                "type": "end",
                "transcript": transcript
            })
    async def process(self, env, event):
        logger.info(
            f"TranscriptWrapper process for {event}. "
        )
        # Other events must also filter the history.
        if event.history is not None:
            event.history = [
                e for e in event.history
                if not isinstance(e, UserCustomSent)
            ]
        # 1. Process User Voice Input (Triggered after automatic STT)
        if isinstance(event, UserTurnEnded):
            for content in event.content:
                if isinstance(content, UserTextSent):
                    yield AgentSendCustom(metadata={
                        "role": "user",
                        "type": "voice",
                        "transcript": content.content
                    })
            # Pass to LLMAgent for processing.
            async for output in self._process_inner(env, event):
                yield output
            return

        # 2. Handling User Text Input (Sent via Custom Events) This section requires optimization.
        if isinstance(event, UserCustomSent):
            if event.metadata.get("type") == "text":
                user_text = event.metadata.get("content", "")

                # Construct a UserTurnEnded event recognizable by LlmAgent.
                fake_event = UserTurnEnded(
                    content=[UserTextSent(content=user_text)],
                    history= event.history,
                )
                async for output in self.inner.process(env, fake_event):
                    yield output
                return

        # 3. Other events (such as CallStarted) are processed normally.
        async for output in self._process_inner(env, event):
            yield output

async def get_agent(env: AgentEnv, call_request: CallRequest):
    logger.info(
        f"Starting new call for {call_request.call_id}. "
        f"Agent system prompt: {call_request.agent.system_prompt}"
        f"Agent introduction: {call_request.agent.introduction}"
    )
    run_filter = [CallStarted, UserTurnEnded, UserCustomSent, CallEnded]
    cancel_filter = [UserTurnStarted]
    config = LlmConfig()
    if call_request.agent.system_prompt:
        config.system_prompt =  call_request.agent.system_prompt
    else:
        config.system_prompt = "You are a helpful assistant."
    if call_request.agent.introduction:
        config.introduction = call_request.agent.introduction
    llm_agent = LlmAgent(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("API_KEY"),
        tools=[web_search],
        config=config,
    )
    return TranscriptWrapper(llm_agent), run_filter, cancel_filter


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("Starting app")
    app.run()
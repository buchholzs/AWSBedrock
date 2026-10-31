import logging
import os
from functools import lru_cache

from strands import Agent
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.tools.mcp import MCPClient
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from model.load import load_model
from parse import parse_payload
from memory.session import get_memory_session_manager

from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Use tools when appropriate."


@lru_cache(maxsize=1)
def _get_gateway_mcp_client() -> MCPClient:
    """Create a single MCP client connected to the AgentCore gateway."""
    endpoint = _get_gateway_endpoint()
    region = os.getenv("AGENTCORE_GATEWAY_REGION") or os.getenv("AWS_REGION") or "eu-central-1"

    client = MCPClient(
        lambda: aws_iam_streamablehttp_client(
            endpoint=endpoint,
            aws_service="bedrock-agentcore",
            aws_region=region,
        )
    )
    client.start()
    return client

def _make_conversation_manager():
    return NullConversationManager()


def _get_gateway_endpoint() -> str:
    endpoint = os.getenv("AGENTCORE_GATEWAY_ENDPOINT") or os.getenv("BEDROCK_AGENTCORE_GATEWAY_ENDPOINT")
    if not endpoint:
        raise RuntimeError(
            "Missing required AgentCore gateway endpoint. Set AGENTCORE_GATEWAY_ENDPOINT or "
            "BEDROCK_AGENTCORE_GATEWAY_ENDPOINT."
        )
    return endpoint


@lru_cache(maxsize=128)
def _get_agent(session_id: str, actor_id: str) -> Agent:
    """Given a session_id and actor_id, return or construct the corresponding Strands agent.
    Note: caching helps avoid repeated identity fetches on non-bedrock model loads
    and supports in-memory session management for local dev."""
    gateway_client = _get_gateway_mcp_client()
    return Agent(
        model=load_model(),
        session_manager=get_memory_session_manager(session_id, actor_id),
        conversation_manager=_make_conversation_manager(),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        tools=list(gateway_client.list_tools_sync()),
    )

def create_app():
    app = BedrockAgentCoreApp()
    log = app.logger

    @app.entrypoint
    async def invoke(payload, context):
        log.info("Invoking Agent.....")

        session_id = getattr(context, "session_id", None) or "default-session"
        prompt, actor_id = parse_payload(payload)

        log.info(f"Invoking with session_id={session_id} and actor_id={actor_id}")
        agent = _get_agent(session_id, actor_id)

        async for event in agent.stream_async(prompt):
            if not isinstance(event, dict) or "event" not in event:
                continue
            cbs = event["event"].get("contentBlockStart")
            if cbs is not None and not cbs.get("start"):
                continue
            yield event

    return app

app = create_app()
if __name__ == "__main__":
    app.run()

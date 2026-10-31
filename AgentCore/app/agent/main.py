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
from bedrock_agentcore.gateway.integrations.strands.plugins import AgentCoreToolSearchPlugin

logger = logging.getLogger(__name__)


class GatewayToolSearchPlugin(AgentCoreToolSearchPlugin):
    """Use semantic search when available, but fall back to direct MCP tool listing."""

    def on_before_invocation(self, event):
        messages = event.messages or []
        intent = self._intent_provider.derive_intent(messages, model=event.agent.model)
        logger.info("Derived intent: %s", intent)

        for name in list(self._loaded_tool_names):
            event.agent.tool_registry.registry.pop(name, None)
        self._loaded_tool_names.clear()

        if not intent:
            return

        try:
            result = self._mcp_client.call_tool_sync(
                tool_use_id="intent-search",
                name="x_amz_bedrock_agentcore_search",
                arguments={"query": intent},
            )
            agent_tools = self._build_tools_from_search_result(result)
        except Exception as exc:  # pragma: no cover - safety fallback
            logger.warning("AgentCore semantic search failed for intent %s: %s", intent, exc)
            agent_tools = []

        if not agent_tools:
            logger.warning(
                "AgentCore semantic search returned no tools for intent %q. Falling back to direct gateway tool discovery.",
                intent,
            )
            try:
                agent_tools = list(self._mcp_client.list_tools_sync())
            except Exception as exc:  # pragma: no cover - safety fallback
                logger.error("Gateway fallback tool discovery failed: %s", exc)
                return

        for agent_tool in agent_tools:
            try:
                if (
                    agent_tool.tool_name in event.agent.tool_registry.registry
                    and agent_tool.tool_name not in self._loaded_tool_names
                ):
                    logger.debug(
                        "Skipping tool %s: already registered as a static tool",
                        agent_tool.tool_name,
                    )
                    continue
                event.agent.tool_registry.register_tool(agent_tool)
                self._loaded_tool_names.add(agent_tool.tool_name)
            except Exception as exc:
                logger.error("Failed to register tool %s: %s", agent_tool.tool_name, exc)

        logger.info("Loaded tools: %s", self._loaded_tool_names)

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
        plugins=[GatewayToolSearchPlugin(mcp_client=gateway_client)],
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

import os

from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

MEMORY_ID = os.getenv("BEDROCK_AGENTCORE_MEMORY_ID") or os.getenv("MEMORY_AGENTMEMORY_ID")
REGION = os.getenv("AWS_REGION")

def get_memory_session_manager(
        session_id: str, actor_id: str,
) -> AgentCoreMemorySessionManager:
    if not MEMORY_ID:
        raise RuntimeError(
            "AgentCore Memory is not configured. Set BEDROCK_AGENTCORE_MEMORY_ID to the memory resource ID."
        )

    retrieval_config = {
        f"/users/{actor_id}/facts": RetrievalConfig(top_k=3, relevance_score=0.5),
        f"/users/{actor_id}/preferences": RetrievalConfig(top_k=3, relevance_score=0.5),
        f"/episodes/{actor_id}/{session_id}": RetrievalConfig(top_k=5, relevance_score=0.5),
        f"/summaries/{actor_id}/{session_id}": RetrievalConfig(top_k=3, relevance_score=0.5),
    }

    return AgentCoreMemorySessionManager(
        AgentCoreMemoryConfig(
            memory_id=MEMORY_ID,
            session_id=session_id,
            actor_id=actor_id,
            retrieval_config=retrieval_config,
        ),
        REGION,
    )

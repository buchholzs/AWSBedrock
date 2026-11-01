import os

from strands.models.bedrock import BedrockModel


def load_model() -> BedrockModel:
    """Get the Bedrock model client with the configured Guardrail."""
    guardrail_id = os.getenv("BEDROCK_GUARDRAIL_ID")
    guardrail_version = os.getenv("BEDROCK_GUARDRAIL_VERSION")
    if not guardrail_id or not guardrail_id.strip():
        raise RuntimeError("Missing required Bedrock Guardrail configuration: BEDROCK_GUARDRAIL_ID")
    if not guardrail_version or not guardrail_version.strip():
        raise RuntimeError("Missing required Bedrock Guardrail configuration: BEDROCK_GUARDRAIL_VERSION")

    return BedrockModel(
        model_id="eu.anthropic.claude-sonnet-4-6",
        guardrail_id=guardrail_id.strip(),
        guardrail_version=guardrail_version.strip(),
    )

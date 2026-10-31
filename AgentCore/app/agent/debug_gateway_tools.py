import json
import os

import httpx

from strands.tools.mcp import MCPClient

from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client


def get_gateway_endpoint() -> str:
    endpoint = os.getenv("AGENTCORE_GATEWAY_ENDPOINT") or os.getenv("BEDROCK_AGENTCORE_GATEWAY_ENDPOINT")
    if not endpoint:
        raise RuntimeError(
            "Missing AGENTCORE_GATEWAY_ENDPOINT or BEDROCK_AGENTCORE_GATEWAY_ENDPOINT. "
            "Set it before running this check."
        )
    return endpoint


def main() -> None:
    endpoint = get_gateway_endpoint()
    region = os.getenv("AGENTCORE_GATEWAY_REGION") or os.getenv("AWS_REGION") or "eu-central-1"

    print(f"Gateway endpoint: {endpoint}")
    print(f"Gateway region: {region}")

    print("\nRaw HTTP probe to the gateway...")
    try:
        from botocore.session import Session as BotoSession

        creds = BotoSession().get_credentials()
        if creds is None:
            raise RuntimeError("No AWS credentials available for SigV4 signing.")

        auth = httpx.Auth(
            lambda request: request
        )
        del auth

        # Use the same AWS SigV4 signing approach as the mcp client by leveraging the auth helper.
        from mcp_proxy_for_aws.sigv4_helper import SigV4HTTPXAuth

        auth = SigV4HTTPXAuth(creds, "bedrock-agentcore", region)
        with httpx.Client(auth=auth, timeout=30.0) as http_client:
            req = httpx.Request(
                "POST",
                endpoint,
                headers={"accept": "application/json, text/event-stream"},
                content=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}),
            )
            prepared = http_client.build_request(req)
            response = http_client.send(prepared)
            print("HTTP status:", response.status_code)
            print("HTTP headers:")
            for k, v in response.headers.items():
                print(f"  {k}: {v}")
            print("HTTP body:")
            print(response.text[:8000])
    except Exception as exc:
        print(f"RAW_HTTP_ERROR: {type(exc).__name__}: {exc}")

    client = MCPClient(
        lambda: aws_iam_streamablehttp_client(
            endpoint=endpoint,
            aws_service="bedrock-agentcore",
            aws_region=region,
        )
    )

    client.start()
    print("MCP client started")

    try:
        print("\nListing gateway tools...")
        try:
            tool_list = client.list_tools_sync()
            tools = list(tool_list)
            print(f"tool_count={len(tools)}")

            if not tools:
                print("RESULT: no tools are registered on the gateway")
            else:
                for tool in tools:
                    description = getattr(tool.mcp_tool, "description", None) or "(no description)"
                    print(f"- {tool.tool_name}: {description[:200]}")
        except Exception as exc:
            print(f"LIST_TOOLS_ERROR: {type(exc).__name__}: {exc}")

        print("\nTrying semantic search on x_amz_bedrock_agentcore_search...")
        try:
            result = client.call_tool_sync(
                tool_use_id="gateway-diagnostic-search",
                name="x_amz_bedrock_agentcore_search",
                arguments={"query": "show me tools relevant to this gateway"},
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str)[:4000])
        except Exception as exc:
            print(f"SEARCH_ERROR: {type(exc).__name__}: {exc}")

    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

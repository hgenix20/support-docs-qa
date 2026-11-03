import json
from typing import Any

from aws_cdk import App
from aws_cdk.assertions import Template

from infra.stack import SupportDocsStack

EMBED = "amazon.titan-embed-text-v2:0"
GENERATION = "test.generation-model-v1:0"


def _template() -> Template:
    app = App()
    stack = SupportDocsStack(app, "Test", embed_model_id=EMBED, generation_model_id=GENERATION)
    return Template.from_stack(stack)


def test_three_tables_and_query_log_has_ttl() -> None:
    template = _template()
    template.resource_count_is("AWS::DynamoDB::Table", 3)
    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {"TimeToLiveSpecification": {"AttributeName": "expires_at", "Enabled": True}},
    )


def test_chunks_table_composite_key() -> None:
    template = _template()
    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "KeySchema": [
                {"AttributeName": "doc_id", "KeyType": "HASH"},
                {"AttributeName": "chunk_index", "KeyType": "RANGE"},
            ]
        },
    )


def test_both_functions_use_python_312_and_expected_handlers() -> None:
    template = _template()
    for handler in ("handlers.ingest.handler", "handlers.ask.handler"):
        template.has_resource_properties(
            "AWS::Lambda::Function", {"Handler": handler, "Runtime": "python3.12"}
        )


def test_every_api_method_requires_a_key() -> None:
    methods = _template().find_resources("AWS::ApiGateway::Method")
    post_methods = [m for m in methods.values() if m["Properties"]["HttpMethod"] == "POST"]
    assert len(post_methods) == 2
    assert all(m["Properties"]["ApiKeyRequired"] is True for m in post_methods)


def test_bedrock_permissions_are_scoped_to_named_models() -> None:
    policies = _template().find_resources("AWS::IAM::Policy")
    bedrock_statements: list[dict[str, Any]] = []
    for policy in policies.values():
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            actions = statement["Action"]
            if actions == "bedrock:InvokeModel" or actions == ["bedrock:InvokeModel"]:
                bedrock_statements.append(statement)
    assert bedrock_statements, "expected Bedrock invoke statements"
    for statement in bedrock_statements:
        assert statement["Resource"] != "*"
        rendered = json.dumps(statement["Resource"])
        assert "foundation-model/" in rendered or "inference-profile/" in rendered


def test_no_stray_wildcard_resources() -> None:
    policies = _template().find_resources("AWS::IAM::Policy")
    for policy in policies.values():
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            assert statement.get("Resource") != "*"

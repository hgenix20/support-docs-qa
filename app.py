"""CDK app entry point. Model IDs come from cdk.json context and can be overridden with -c."""

import aws_cdk as cdk

from infra.stack import SupportDocsStack

PLACEHOLDER_PREFIX = "REPLACE-WITH"

app = cdk.App()
generation_model_id = app.node.get_context("generation_model_id")
if not isinstance(generation_model_id, str) or generation_model_id.startswith(PLACEHOLDER_PREFIX):
    raise SystemExit(
        "generation_model_id is unset. Set it in cdk.json or pass -c generation_model_id=<id>. "
        "List available IDs with: aws bedrock list-foundation-models --by-provider anthropic"
    )

SupportDocsStack(
    app,
    "SupportDocsQaStack",
    embed_model_id=app.node.get_context("embed_model_id"),
    generation_model_id=generation_model_id,
)
app.synth()

"""CDK stack: DynamoDB tables, two Lambdas, and a REST API protected by an API key.

Least privilege: ingest may only invoke the embedding model; ask may invoke the
embedding and generation models. Each Lambda gets only the table access it uses.
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    Aws,
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_apigateway as apigw,
)
from aws_cdk import (
    aws_dynamodb as ddb,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_lambda as _lambda,
)
from aws_cdk import (
    aws_logs as logs,
)
from constructs import Construct

# Lambda code is the whole src/ tree; handlers import the rag package from it.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"

MAX_SCAN_CHUNKS = "2000"
QUERY_TTL_DAYS = "30"


class SupportDocsStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        embed_model_id: str,
        generation_model_id: str,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)  # type: ignore[arg-type]

        # Showcase data is disposable, so tables are destroyed with the stack.
        # For real data, switch to RemovalPolicy.RETAIN.
        chunks = ddb.Table(
            self,
            "ChunksTable",
            partition_key=ddb.Attribute(name="doc_id", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="chunk_index", type=ddb.AttributeType.NUMBER),
            billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )
        documents = ddb.Table(
            self,
            "DocumentsTable",
            partition_key=ddb.Attribute(name="doc_id", type=ddb.AttributeType.STRING),
            billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )
        query_log = ddb.Table(
            self,
            "QueryLogTable",
            partition_key=ddb.Attribute(name="query_id", type=ddb.AttributeType.STRING),
            billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="expires_at",
            removal_policy=RemovalPolicy.DESTROY,
        )

        embed_arn = f"arn:{Aws.PARTITION}:bedrock:{Aws.REGION}::foundation-model/{embed_model_id}"
        # Newer Claude models are invoked through inference profiles, which route to
        # other regions. Cover both the profile and the foundation model in any region.
        generation_arns = [
            f"arn:{Aws.PARTITION}:bedrock:*::foundation-model/{generation_model_id}",
            f"arn:{Aws.PARTITION}:bedrock:{Aws.REGION}:{Aws.ACCOUNT_ID}:inference-profile/{generation_model_id}",
        ]

        shared_env = {
            "CHUNKS_TABLE": chunks.table_name,
            "DOCUMENTS_TABLE": documents.table_name,
            "QUERY_LOG_TABLE": query_log.table_name,
            "EMBED_MODEL_ID": embed_model_id,
            "GENERATION_MODEL_ID": generation_model_id,
            "MAX_SCAN_CHUNKS": MAX_SCAN_CHUNKS,
        }

        ingest_fn = self._function("IngestFunction", "handlers.ingest.handler", shared_env)
        ask_fn = self._function("AskFunction", "handlers.ask.handler", shared_env)

        chunks.grant_read_write_data(ingest_fn)
        documents.grant_write_data(ingest_fn)
        ingest_fn.add_to_role_policy(
            iam.PolicyStatement(actions=["bedrock:InvokeModel"], resources=[embed_arn])
        )

        chunks.grant_read_data(ask_fn)
        query_log.grant_write_data(ask_fn)
        ask_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"], resources=[embed_arn, *generation_arns]
            )
        )

        api = apigw.RestApi(
            self,
            "SupportDocsApi",
            rest_api_name="support-docs-qa",
            deploy_options=apigw.StageOptions(
                stage_name="v1",
                throttling_rate_limit=5,
                throttling_burst_limit=10,
            ),
        )
        api_key = api.add_api_key("ClientKey")
        plan = api.add_usage_plan(
            "UsagePlan",
            throttle=apigw.ThrottleSettings(rate_limit=5, burst_limit=10),
            quota=apigw.QuotaSettings(limit=1000, period=apigw.Period.DAY),
        )
        plan.add_api_key(api_key)
        plan.add_api_stage(stage=api.deployment_stage)

        api.root.add_resource("documents").add_method(
            # jsii protocol false positive: Function is an IFunction at runtime, not to pyright.
            "POST",
            apigw.LambdaIntegration(ingest_fn),  # type: ignore[arg-type]
            api_key_required=True,
        )
        api.root.add_resource("ask").add_method(
            "POST",
            apigw.LambdaIntegration(ask_fn),  # type: ignore[arg-type]
            api_key_required=True,
        )

        CfnOutput(self, "ApiUrl", value=api.url)
        CfnOutput(
            self,
            "ApiKeyId",
            value=api_key.key_id,
            description=(
                "Fetch the value with: aws apigateway get-api-key --api-key <id> --include-value"
            ),
        )

    def _function(self, construct_id: str, handler: str, env: dict[str, str]) -> _lambda.Function:
        log_group = logs.LogGroup(
            self,
            f"{construct_id}Logs",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )
        return _lambda.Function(
            self,
            construct_id,
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler=handler,
            code=_lambda.Code.from_asset(str(SRC_DIR), exclude=["**/__pycache__"]),
            memory_size=512,
            timeout=Duration.seconds(28),  # below the 29 s API Gateway integration limit
            environment=env,
            log_group=log_group,
        )

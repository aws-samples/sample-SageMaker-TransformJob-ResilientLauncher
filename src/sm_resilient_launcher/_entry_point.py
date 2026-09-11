"""Entry point: launch_create_transform_job.

The single public API for launching SageMaker Batch Transform jobs with
capacity resilience.  Accepts all boto3 ``create_transform_job``
PascalCase parameters plus SDK-specific parameters.

**Default behavior:** When no SDK-specific parameters are provided,
the function behaves identically to ``boto3.client("sagemaker")
.create_transform_job()``.  This makes it a true drop-in replacement.

Return type is ``LaunchResult | dict``:

* ``dict`` — when passthrough (no SDK params, or ``Strategy="passthrough"``).
* ``LaunchResult`` — for all other execution paths.

Requirements covered: 1.4, 1.6, 1.7, 2.1–2.8, 4.1–4.3.
"""

from __future__ import annotations

import boto3

from sm_resilient_launcher._error_classifier import ErrorClassifier
from sm_resilient_launcher._intelligent_router import IntelligentRouter
from sm_resilient_launcher._launch_result import LaunchResult
from sm_resilient_launcher._launcher import Launcher
from sm_resilient_launcher._param_separator import ParamSeparator
from sm_resilient_launcher._pricing_fetcher import PricingFetcher
from sm_resilient_launcher._quota_fetcher import QuotaFetcher
from sm_resilient_launcher._transform_adapter import TransformJobAdapter
from sm_resilient_launcher._type_expander import TypeExpander


def launch_create_transform_job(**kwargs) -> LaunchResult | dict:
    """Launch a SageMaker Batch Transform job with capacity resilience.

    Accepts every parameter that boto3's ``create_transform_job``
    accepts (PascalCase names), plus SDK-specific parameters:

    * ``Strategy`` — ``"passthrough"``, ``"sequential"``, or
      ``"race"``.  Omit to let the SDK infer from input shape.
    * ``AllowedRegions`` — list of AWS regions for cross-region
      fallback.
    * ``WorkloadDescription`` — natural-language workload description
      for AI-powered type expansion via Bedrock.
    * ``RegionConfig`` — per-region overrides (ModelName, etc.).
    * ``AutoCreateModel`` — auto-create models in target regions.
    * ``MaxRetries`` — retry attempts on capacity errors.
    * ``BackoffBase`` — seconds before first retry.
    * ``BackoffMultiplier`` — exponential backoff multiplier.

    **Default behavior (no SDK params):**
        When no SDK-specific parameters are provided and InstanceType
        is a scalar string, the function is a direct passthrough to
        boto3 — identical behavior to the native API.  This makes it
        a true drop-in replacement: just change the import.

    **Passthrough path** (``Strategy="passthrough"`` or no SDK params):
        Calls ``boto3.client("sagemaker").create_transform_job()``
        directly with zero SDK logic, zero retry, zero error
        classification, and returns the raw boto3 API response
        ``dict``.

    **All other paths**:
        Instantiates the full SDK component graph (adapter, engines,
        router, expander) and delegates to ``Launcher.launch()``,
        returning a ``LaunchResult``.

    Args:
        **kwargs: All boto3 ``create_transform_job`` PascalCase
            parameters, plus SDK-specific params listed above.

    Returns:
        ``LaunchResult`` for non-passthrough paths, or a raw ``dict``
        for passthrough.

    Raises:
        AllTypesExhaustedError: When all instance types are exhausted
            in a single-region execution.
        AllRegionsExhaustedError: When all regions and types are
            exhausted in cross-region execution.
    """
    # ── 1. Separate SDK params from boto3 params ───────────────
    boto3_params, sdk_params, is_list_mode = ParamSeparator.separate(
        kwargs
    )

    # ── 2. Passthrough: explicit or default (no SDK params) ────
    #
    # Passthrough when:
    #   a) Strategy="passthrough" is explicitly set, OR
    #   b) No SDK params at all AND InstanceType is scalar
    #      (true drop-in replacement for boto3)
    explicit_passthrough = sdk_params.get("Strategy") == "passthrough"
    no_sdk_params = len(sdk_params) == 0
    default_passthrough = no_sdk_params and not is_list_mode

    if explicit_passthrough or default_passthrough:
        session = boto3.session.Session()
        region = session.region_name or "us-east-1"
        client = boto3.client("sagemaker", region_name=region)
        return client.create_transform_job(**boto3_params)

    # ── 3. Extract instance_types list ─────────────────────────
    instance_type_value = boto3_params.get(
        "TransformResources", {}
    ).get("InstanceType")

    if isinstance(instance_type_value, list):
        instance_types = list(instance_type_value)  # preserve order
    else:
        instance_types = (
            [instance_type_value] if instance_type_value else []
        )

    # ── 4. Determine region ────────────────────────────────────
    session = boto3.session.Session()
    region = session.region_name or "us-east-1"

    # If AllowedRegions is provided, use the first entry as the
    # home region (consistent with CrossRegionManager).
    allowed_regions = sdk_params.get("AllowedRegions")
    if allowed_regions:
        region = allowed_regions[0]

    # ── 5. Instantiate component graph ─────────────────────────
    adapter = TransformJobAdapter(region)
    error_classifier = ErrorClassifier

    quota_fetcher = QuotaFetcher(region)
    pricing_fetcher = PricingFetcher()
    type_expander = TypeExpander(quota_fetcher)
    intelligent_router = IntelligentRouter(
        quota_fetcher, pricing_fetcher, type_expander, region
    )

    adapter_factory = TransformJobAdapter  # callable(region) -> adapter

    launcher = Launcher(
        adapter=adapter,
        error_classifier=error_classifier,
        intelligent_router=intelligent_router,
        type_expander=type_expander,
        adapter_factory=adapter_factory,
    )

    # ── 6. Delegate to Launcher ────────────────────────────────
    return launcher.launch(
        instance_types=instance_types,
        boto3_params=boto3_params,
        sdk_params=sdk_params,
        region=region,
    )

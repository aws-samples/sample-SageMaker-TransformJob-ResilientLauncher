# SageMaker Capacity-Resilient Job Launcher

Automatic capacity fallback across instance types and AWS regions for SageMaker Batch Transform. One import change from native boto3.

## The Problem

SageMaker's `CreateTransformJob` API accepts a single instance type per request. When that type has no capacity, the job hard-fails in approximately 40 seconds. There is no built-in fallback list, no cross-region retry, and no automatic resubmission. Instance pools (the AWS-native solution) only cover real-time endpoints — Batch Transform and Training Jobs are not covered.

GPU batch workloads require manual resubmission with different instance types until something sticks, and cross-region failover requires manual S3 data replication and job reconfiguration.

## The Solution

This SDK wraps boto3's `create_transform_job` to provide five behaviors — all accessible through a single function that uses the exact same PascalCase parameter names as the native API:

| Behavior | When | What Happens |
|---|---|---|
| **Native Passthrough** | `Strategy="passthrough"` | Direct boto3 call, zero SDK logic, returns raw dict |
| **AI Expansion** | Scalar type, no Strategy | TypeExpander (deterministic) or Bedrock Converse generates a same-or-larger fallback list |
| **Sequential Fallback** | List of types or `Strategy="sequential"` | Types tried in order; error classification drives skip behavior |
| **Race (Parallel)** | `Strategy="race"` | All types submitted concurrently; first to provision wins, losers cancelled |
| **Cross-Region** | `AllowedRegions` with 2+ regions | Falls back to other regions when the home region is exhausted |

Migration from native boto3 requires changing a single import.

## Installation

```bash
# From the repository root:
cd sm-resilient-launcher

# Install for development
pip install -e .

# Install with dev dependencies (pytest, hypothesis)
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.10, `boto3`, `botocore`

## Quick Start

### 1. Native Passthrough — identical to boto3

```python
from sm_resilient_launcher import launch_create_transform_job

result = launch_create_transform_job(
    TransformJobName="my-job",
    ModelName="my-model",
    TransformInput={
        "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "s3://bucket/input"}},
        "ContentType": "text/csv",
    },
    TransformOutput={"S3OutputPath": "s3://bucket/output"},
    TransformResources={"InstanceType": "ml.g5.4xlarge", "InstanceCount": 1},
    Strategy="passthrough",
)
# Returns the raw boto3 dict — zero SDK logic applied.
```

### 2. AI Expansion — SDK generates fallback list

```python
result = launch_create_transform_job(
    TransformJobName="my-job",
    ModelName="my-model",
    TransformInput={...},
    TransformOutput={...},
    TransformResources={"InstanceType": "ml.g4dn.xlarge", "InstanceCount": 1},
)
# No Strategy, no WorkloadDescription — TypeExpander generates same-or-larger types.
# SDK tries ml.g4dn.xlarge first, then g4dn.2xlarge, g5.xlarge, etc.
print(f"Landed on {result.instance_type_used} after {result.attempts} attempts")
```

With Bedrock-powered reasoning:

```python
result = launch_create_transform_job(
    TransformJobName="my-job",
    ModelName="my-model",
    TransformInput={...},
    TransformOutput={...},
    TransformResources={"InstanceType": "ml.g4dn.xlarge", "InstanceCount": 1},
    WorkloadDescription="ResNet-50 inference, fp32, ~200MB GPU memory, cost-sensitive",
)
print(result.agent_reasoning)
# "ResNet-50 fp32 needs ~200MB. All G-family types fit. g4dn.xlarge cheapest.
#  Priority: ml.g4dn.xlarge, ml.g4dn.2xlarge, ml.g5.xlarge."
```

### 3. Sequential Fallback — explicit priority list

```python
result = launch_create_transform_job(
    TransformJobName="my-job",
    ModelName="my-model",
    TransformInput={...},
    TransformOutput={...},
    TransformResources={
        "InstanceType": ["ml.g5.4xlarge", "ml.g4dn.2xlarge", "ml.g4dn.xlarge"],
        "InstanceCount": 1,
    },
    Strategy="sequential",
)
# Tries types in your exact order. Never reorders your list.
```

### 4. Race Mode — parallel submission

```python
result = launch_create_transform_job(
    TransformJobName="my-job",
    ModelName="my-model",
    TransformInput={...},
    TransformOutput={...},
    TransformResources={
        "InstanceType": ["ml.g4dn.xlarge", "ml.g5.xlarge"],
        "InstanceCount": 1,
    },
    Strategy="race",
)
# Both submitted in parallel. First to provision wins.
# Losers are cancelled — jobs that never reach TransformStartTime cost $0.
```

### 5. Cross-Region Resilience

```python
result = launch_create_transform_job(
    TransformJobName="my-job",
    ModelName="my-model",
    TransformInput={...},
    TransformOutput={...},
    TransformResources={"InstanceType": "ml.g4dn.xlarge", "InstanceCount": 1},
    AllowedRegions=["us-east-1", "us-west-2", "eu-west-1"],
    AutoCreateModel=True,  # SDK creates the SageMaker Model in target regions
)
print(f"Landed in {result.region_used}, data_replicated={result.data_replicated}")
```

## Architecture

```
+------------------------------------------------------------------+
|                         Customer Code                            |
|   launch_create_transform_job(Strategy="race", ...)              |
+------------------------------+-----------------------------------+
                               |
+------------------------------v-----------------------------------+
|                        ParamSeparator                            |
|   Splits SDK params (Strategy, AllowedRegions, WorkloadDesc,     |
|   RegionConfig, AutoCreateModel) from boto3 params.              |
|   Detects scalar vs list InstanceType.                           |
+------------------------------+-----------------------------------+
                               |
              +----------------v-----------------+
              |   Strategy Selection (Behavior    |
              |   Matrix -- see table below)      |
              +-+------+------+------+-----------+
                |      |      |      |
    +-----------v--+   |   +--v--+   |
    |  passthrough  |   |   |list |   |
    |  Direct boto3 |   |   |mode |   |
    |  Zero SDK     |   |   +--+--+   |
    +---------------+   |      |      |
               +--------v------v--+   |
               |   AI Expansion   |   |
               |  (scalar mode)   |   |
               +--+------------+--+   |
                  |            |      |
     +------------v--+  +-----v------v------------------+
     |Intelligent    |  | TypeExpander                   |
     |Router         |  | Deterministic GPU metadata     |
     |Bedrock + Quota|  | Same-or-larger expansion       |
     |+ Pricing      |  +------------------+-------------+
     +-------+-------+                     |
             +---------+---------+---------+
                       |
          +------------v-------------+
          |         Launcher          |
          |  Job-type-agnostic        |
          |  orchestrator             |
          +---+--------------+-------+
              |              |
   +----------v---+   +-----v--------------------+
   | Sequential   |   | RaceEngine               |
   | Engine       |   | Parallel submit           |
   | Ordered retry|   | Winner detection          |
   | by error     |   | Loser cancellation        |
   +------+-------+   +-----+--------------------+
          |                  |
          +--------+---------+
                   |
      +------------v--------------------------+
      |       CrossRegionManager               |
      |  S3 mirror buckets, ECR warning,       |
      |  IAM validate, AutoCreateModel,        |
      |  data replication, output retrieval,    |
      |  SDK tags                              |
      +------------+--------------------------+
                   |
      +------------v--------------------------+
      |   JobAdapter (abstract base class)     |
      |  create_job, describe_job, stop_job    |
      +------------+--------------------------+
                   |
      +------------v--------------------------+
      |   TransformJobAdapter (concrete)       |
      |  CreateTransformJob,                   |
      |  DescribeTransformJob,                 |
      |  StopTransformJob                      |
      +------------+--------------------------+
                   |
      +------------v--------------------------+
      |         ErrorClassifier                |
      |  capacity | quota | family_support     |
      |  | unknown                             |
      +-------------------------------------------+
```

## Behavior Matrix

The SDK infers the correct behavior from the shape of your input:

| InstanceType Shape | Strategy Param | AllowedRegions | Behavior |
|---|---|---|---|
| Scalar string | absent | absent or 1 region | **AI Expansion** then sequential. Bedrock if `WorkloadDescription` present, TypeExpander otherwise. |
| Scalar string | absent | 2+ regions | **Cross-region** with single type, sequential per region. |
| Scalar string | explicit (not passthrough) | any | Execute specified strategy with the single type. |
| Scalar string | `"passthrough"` | any | **Native passthrough.** Direct boto3. Zero SDK logic. |
| List | absent | any | **Sequential** with caller's list in exact order. |
| List | explicit | any | Execute specified strategy with caller's list in exact order. |

**Invariant:** The SDK never reorders, removes from, or modifies a caller-provided instance priority list.

## Strategy Selection Logic

1. **Passthrough** (`Strategy="passthrough"`): Handled first. The request is forwarded directly to boto3 with zero SDK logic, zero retry, zero error classification. Returns the raw boto3 API response as a `dict`.

2. **Scalar input, no Strategy**: The SDK expands the single instance type into a fallback list. If `WorkloadDescription` is provided, the `IntelligentRouter` calls Bedrock Converse with deterministic quota/pricing facts to reason about GPU memory fit. Otherwise, the `TypeExpander` generates the list using hardcoded GPU metadata (same-or-larger types, filtered by non-zero quota). The expanded list is executed via sequential fallback.

3. **List input or explicit Strategy**: The caller's list is used in exact order. No AI expansion, no reordering. The specified (or defaulted) strategy engine executes the list.

4. **Cross-region**: When `AllowedRegions` contains 2+ regions, the `CrossRegionManager` tries each region in order. Pre-flight checks, data replication, and output retrieval are handled automatically.

## Cross-Region Deep Dive

Cross-region fallback is the most complex feature. This section covers everything you need to know.

### What needs to exist in each target region

| Resource | Home Region | Target Region | Who Creates It |
|---|---|---|---|
| S3 input data | Your bucket | Mirror bucket `{bucket}-{region}` | SDK auto-creates bucket, auto-copies data |
| S3 model artifact | Your bucket | Mirror bucket | SDK auto-copies |
| S3 output path | Your bucket | Mirror bucket | SDK copies output back after completion |
| SageMaker Model | Your model | Needs a model in target region | See three options below |
| Container image (ECR) | Your repo | Needs image in target ECR | You push it, or AWS-managed images resolve automatically |
| IAM execution role | Your role | Same role (IAM is global) | Nothing to do |

### Three options for cross-region model resolution

#### Option A: `RegionConfig` — explicit model name per region (safest)

Pre-create your SageMaker Model in each target region (or use `prepare_regions` to check readiness), then tell the SDK which model name to use per region:

```python
result = launch_create_transform_job(
    TransformJobName="my-job",
    ModelName="my-model",  # used in us-east-1
    TransformInput={...},
    TransformOutput={...},
    TransformResources={"InstanceType": "ml.g5.4xlarge", "InstanceCount": 1},
    AllowedRegions=["us-east-1", "us-west-2", "eu-west-1"],
    RegionConfig={
        "us-west-2": {"ModelName": "my-model-usw2"},
        "eu-west-1": {"ModelName": "my-model-euw1"},
    },
)
```

When `RegionConfig` provides a `ModelName` for a region, the SDK:
- Skips the ECR image check for that region (trusts your setup).
- Skips model artifact replication (your model already points to the right artifact).
- Uses the specified model name when submitting the job.

This is the safest option because you control every piece of the target-region setup.

#### Option B: `AutoCreateModel=True` — SDK auto-creates the model

The SDK describes your home-region model, resolves the container image for the target region, rewrites the model artifact S3 URI to the mirror bucket, and calls `CreateModel` in the target region:

```python
result = launch_create_transform_job(
    TransformJobName="my-job",
    ModelName="my-model",
    TransformInput={...},
    TransformOutput={...},
    TransformResources={"InstanceType": "ml.g5.4xlarge", "InstanceCount": 1},
    AllowedRegions=["us-east-1", "us-west-2"],
    AutoCreateModel=True,
)
# SDK creates "my-model-us-west-2" in us-west-2 with resolved image URI.
```

The auto-created model:
- Name: `{original_name}-{target_region}` to avoid collisions.
- Image: Resolved for the target region (see container image scenarios below).
- ModelDataUrl: Rewritten to mirror bucket S3 path.
- ExecutionRoleArn: Same (IAM is global).
- Environment: Preserved from the source model.
- Tags: SDK tags applied.

Created models are cached per region — no re-creation on retry.

#### Option C: Manual setup with `prepare_regions` utility

Run the readiness check before launching jobs:

```python
from sm_resilient_launcher import prepare_regions

report = prepare_regions(
    home_region="us-east-1",
    target_regions=["us-west-2", "eu-west-1"],
    model_name="my-model",
    role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
    auto_create_model=True,  # also check auto-creation feasibility
)

for region in report.regions:
    print(f"{region.region}: S3={region.s3_status}, ECR={region.ecr_status}, IAM={region.iam_status}")
    if region.ecr_commands:
        print(f"  Run: {region.ecr_commands}")
```

### Container image scenarios

How `AutoCreateModel` resolves the target-region container image depends on who owns the image:

| Image Type | Detection | Resolution |
|---|---|---|
| **AWS-managed algorithms** (XGBoost, BlazingText, etc.) | Image account ID differs from your account | SDK uses a hardcoded mapping of SageMaker algorithm account IDs per region to resolve the correct image URI |
| **Customer-owned ECR images** | Image account ID matches your account | SDK replaces the region in the URI (e.g., `us-east-1` to `us-west-2`), then verifies the image exists in target ECR. If missing, warns with exact CLI commands and skips the region |
| **Non-ECR images** (Docker Hub, etc.) | No `.ecr.` in the URI | Not supported for cross-region. SDK warns and skips the region |

AWS-managed algorithm account IDs are maintained for 15 regions: us-east-1, us-east-2, us-west-1, us-west-2, eu-west-1, eu-west-2, eu-central-1, ap-northeast-1, ap-northeast-2, ap-southeast-1, ap-southeast-2, ap-south-1, ca-central-1, eu-north-1, sa-east-1.

### S3 data handling

| Data | Direction | Handling |
|---|---|---|
| **Input data** | Home -> Target | Auto-copied to mirror bucket `{bucket}-{target-region}` before job submission |
| **Model artifact** | Home -> Target | Auto-copied (unless `RegionConfig` provides a target model or `AutoCreateModel` is used) |
| **Job output** | Target -> Home | Auto-retrieved back to home region after job completion |

Mirror bucket naming: `{original-bucket}-{target-region}`. Created automatically if it doesn't exist.

### SDK tags on all created resources

All resources created by the SDK are tagged:

| Tag | Value |
|---|---|
| `sm-resilient-launcher:managed` | `true` |
| `sm-resilient-launcher:source-region` | Home region |
| `sm-resilient-launcher:job-name` | Base job name |
| `sm-resilient-launcher:created-at` | ISO 8601 timestamp |

Tags are applied at creation/copy time, never retroactively. The SDK never modifies tags on resources it did not create.

### KMS-encrypted S3 objects

Cross-region copy of KMS-encrypted objects requires the KMS key to be accessible in the target region. If your S3 objects are encrypted with a customer-managed KMS key, you must either:

- Use a multi-region KMS key, or
- Create a new KMS key in the target region and re-encrypt the data.

The SDK does not handle KMS key management. If the cross-region S3 copy fails due to KMS access issues, the SDK skips the region and moves to the next one.

### VPC-configured models

VPC subnets and security groups are regional. If your SageMaker Model is configured with a VPC, the target-region model needs its own VPC configuration (subnets and security groups in the target region). `AutoCreateModel` does not copy VPC settings. Use `RegionConfig` with pre-configured models for VPC-dependent workloads.

### Multi-container models (inference pipelines)

Multi-container inference pipelines are not supported in the current version. `AutoCreateModel` only handles `PrimaryContainer`. If your model uses `Containers` (inference pipeline), use `RegionConfig` with pre-created models instead.

## Error Classification

Every SageMaker API error is classified into exactly one of four categories using case-insensitive marker matching (checked in priority order):

| Category | Markers | SDK Behavior |
|---|---|---|
| `capacity` | `CapacityError`, `InsufficientInstanceCapacity`, `unable to provision` | Retry with next type or region |
| `quota` | `ResourceLimitExceeded`, `account-level service limit` | Skip type in current region |
| `family_support` | `not supported`, `not one of the supported values` | Skip type in **all** regions |
| `unknown` | No markers match | Skip type, try next |

## prepare_regions Utility

Standalone utility for one-time cross-region infrastructure setup:

```python
from sm_resilient_launcher import prepare_regions

report = prepare_regions(
    home_region="us-east-1",
    target_regions=["us-west-2", "eu-west-1"],
    model_name="my-model",
    role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
    auto_create_model=True,
)

for region in report.regions:
    print(f"{region.region}: S3={region.s3_status}, ECR={region.ecr_status}, IAM={region.iam_status}")
    if region.ecr_commands:
        print(f"  Run: {region.ecr_commands}")
    if region.iam_details:
        print(f"  Warning: {region.iam_details}")
```

Returns a `ReadinessReport` with per-region `RegionReadiness` entries:
- `s3_status`: `"ready"` (existing) or `"created"` (new bucket)
- `ecr_status`: `"ready"` or `"missing"` (with CLI replication commands)
- `iam_status`: `"valid"` or `"warning"` (with policy gap details)

When `auto_create_model=True`, the utility additionally checks whether auto-creation would succeed in each target region (image exists in target ECR for customer images, algorithm account mapping exists for AWS-managed images).

## API Reference

### launch_create_transform_job(\*\*kwargs)

Accepts all boto3 `create_transform_job` PascalCase parameters, plus:

| SDK Parameter | Type | Default | Description |
|---|---|---|---|
| `Strategy` | `str` | Inferred | `"passthrough"`, `"sequential"`, or `"race"`. Omit to let the SDK infer from input shape. |
| `AllowedRegions` | `list[str]` | None | AWS regions for cross-region fallback. First region is the home region. |
| `WorkloadDescription` | `str` | None | Natural-language workload description for Bedrock-powered AI expansion. |
| `RegionConfig` | `dict` | None | Per-region configuration. Keys are region names, values are dicts with optional `ModelName`. |
| `AutoCreateModel` | `bool` | `False` | When `True`, the SDK auto-creates a SageMaker Model in target regions during cross-region fallback. |

`TransformResources.InstanceType` is enhanced to accept either a `str` (scalar) or `list[str]` (priority list).

**Returns:** `LaunchResult` for non-passthrough paths, or a raw `dict` for passthrough.

**Raises:** `AllTypesExhaustedError`, `AllRegionsExhaustedError`

Unrecognized parameters are passed through to boto3 unchanged — the SDK never rejects unknown keys and remains forward-compatible with new API parameters.

### LaunchResult Fields

| Field | Type | Description |
|---|---|---|
| `job_name` | `str` | The winning SageMaker transform job name |
| `instance_type_used` | `str` | The instance type that succeeded |
| `region_used` | `str` | The AWS region where the job ran |
| `attempts` | `int` | Total number of job submission attempts |
| `strategy_used` | `str` | `"sequential"` or `"race"` |
| `agent_reasoning` | `str \| None` | Bedrock reasoning chain, or `None` for deterministic paths |
| `data_replicated` | `bool` | `True` if cross-region data replication occurred |
| `errors` | `list[ClassifiedError]` | All errors from failed attempts (type, region, category) |
| `warnings` | `list[str]` | Pre-flight warnings (ECR commands, IAM advisories) |

### ReadinessReport Fields

| Field | Type | Description |
|---|---|---|
| `regions` | `list[RegionReadiness]` | Per-region readiness entries |

### RegionReadiness Fields

| Field | Type | Description |
|---|---|---|
| `region` | `str` | AWS region name |
| `s3_status` | `str` | `"ready"` or `"created"` |
| `ecr_status` | `str` | `"ready"` or `"missing"` |
| `iam_status` | `str` | `"valid"` or `"warning"` |
| `ecr_commands` | `str \| None` | CLI commands to replicate ECR image (when missing) |
| `iam_details` | `str \| None` | IAM warning details (when applicable) |

## Error Handling

```python
from sm_resilient_launcher import (
    launch_create_transform_job,
    AllTypesExhaustedError,
    AllRegionsExhaustedError,
)

try:
    result = launch_create_transform_job(...)
except AllTypesExhaustedError as e:
    # All instance types failed in a single region
    for err in e.errors:
        print(f"  {err.instance_type} in {err.region}: {err.category.value} - {err.message}")
except AllRegionsExhaustedError as e:
    # All regions and all types exhausted
    regions = sorted({err.region for err in e.errors})
    types = sorted({err.instance_type for err in e.errors})
    print(f"  Exhausted regions: {regions}, types: {types}")
```

## Extensibility — JobAdapter ABC

The SDK uses an abstract `JobAdapter` base class so that Batch Transform-specific logic is isolated in `TransformJobAdapter`. The Launcher, SequentialEngine, RaceEngine, and CrossRegionManager interact with jobs exclusively through this interface.

To add Training Job support, implement `TrainingJobAdapter` and a new `launch_create_training_job` entry point. No modifications needed to the strategy engines, cross-region manager, or orchestration logic.

```python
from sm_resilient_launcher._job_adapter import JobAdapter

class TrainingJobAdapter(JobAdapter):
    def create_job(self, job_name, instance_type, boto3_params): ...
    def describe_job(self, job_name): ...
    def stop_job(self, job_name): ...
    def is_completed(self, description): ...
    def is_failed(self, description): ...
    def extract_error(self, description): ...
```

## Edge Cases and Limitations

- **KMS-encrypted S3 objects**: Cross-region copy needs the KMS key to be accessible in the target region. Use multi-region keys or create target-region keys. The SDK skips the region on copy failure.

- **VPC-configured models**: Subnets and security groups are regional. `AutoCreateModel` does not copy VPC settings. Use `RegionConfig` with pre-configured models.

- **Multi-container inference pipelines**: Not supported in the current version. `AutoCreateModel` handles `PrimaryContainer` only. Use `RegionConfig` for pipeline models.

- **Custom environment variables**: Preserved by `AutoCreateModel`. The SDK copies all `Environment` key-value pairs from the source model to the auto-created model.

- **MaxPayloadInMB x MaxConcurrentTransforms**: Must be <= 100MB. This is a SageMaker service limit, not an SDK limit. The SDK passes these parameters through unchanged.

- **Batch Transform invocation timeout**: Default 600 seconds. Configurable via `ModelClientConfig.InvocationsTimeoutInSeconds`. Passed through unchanged.

- **Large input datasets**: S3 cross-region copy time scales with dataset size. For multi-GB datasets, consider pre-replicating data with S3 Cross-Region Replication (CRR) and using `RegionConfig`.

- **Concurrent job quotas**: Race mode submits N jobs simultaneously. If your account's concurrent transform job quota is low, race mode may hit the limit. Check your quota with `aws service-quotas get-service-quota`.

- **ECR replication**: For frequently used cross-region workloads, set up ECR replication rules rather than manually pushing images. The SDK never replicates ECR images automatically.

- **SageMaker Model name collisions**: `AutoCreateModel` uses the pattern `{model_name}-{region}`. If a model with that name already exists in the target region, the `CreateModel` call will fail and the SDK will skip that region.

- **SDK tag cleanup**: The SDK tags all created resources but does not clean them up after jobs complete. Use tag-based S3 lifecycle policies or manual cleanup for mirror buckets and copied objects.

## Billing Model

Batch Transform does not support spot instances. Billing is per-second from `TransformStartTime` with a 1-minute minimum.

- Jobs that never reach `TransformStartTime` cost **$0**. This makes race mode safe — losing jobs that fail before provisioning are free.
- Stopping a billable non-winner limits waste to at most 1 minute of compute.
- Cross-region S3 data transfer: ~$0.02/GB.

## Development

### Running Unit Tests

```bash
cd sm-resilient-launcher

# Run all 243 unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=sm_resilient_launcher --cov-report=term-missing

# Run a specific test file
pytest tests/test_sequential_engine.py -v

# Run a specific test class
pytest tests/test_error_classifier.py::TestCapacityMarkers -v
```

All unit tests use fully mocked boto3 clients with zero real AWS API calls.

### Integration Testing (Real AWS)

The integration test script runs real Batch Transform jobs against an AWS account:

```bash
# Run all 7 integration tests
python tests/integration/test_real_aws.py

# Run specific tests by number
python tests/integration/test_real_aws.py 1 3 7
```

**Prerequisites:**
- AWS credentials for the target account
- Region us-east-1 configured
- The test model and S3 bucket must exist

**Tests:**
1. Passthrough mode (baseline) — verifies the model works
2. Sequential fallback — quota=0 type fails, falls back to available type
3. Race mode — two types submitted in parallel
4. AI expansion — TypeExpander generates same-or-larger types
5. All types exhausted — verifies `AllTypesExhaustedError`
6. Multi-region fallback — cross-region code path
7. `prepare_regions` utility — ReadinessReport validation

Estimated cost per run: ~$3-5 (each GPU job runs 1-2 minutes).

### Project Structure

```
sm-resilient-launcher/
  pyproject.toml
  README.md
  src/sm_resilient_launcher/
    __init__.py                 # Public API exports
    _entry_point.py             # launch_create_transform_job
    _param_separator.py         # Split SDK/boto3 params
    _launcher.py                # Behavior matrix orchestrator
    _error_classifier.py        # 4-category error classification
    _sequential_engine.py       # Ordered fallback strategy
    _race_engine.py             # Parallel submission strategy
    _intelligent_router.py      # Bedrock Converse AI expansion
    _type_expander.py           # Deterministic GPU metadata expansion
    _quota_fetcher.py           # Service Quotas API
    _pricing_fetcher.py         # Pricing API
    _cross_region_manager.py    # Multi-region fallback + pre-flight + AutoCreateModel
    _job_adapter.py             # Abstract JobAdapter ABC
    _transform_adapter.py       # Batch Transform concrete adapter
    _launch_result.py           # LaunchResult dataclass
    _prepare_regions.py         # prepare_regions utility
    _exceptions.py              # AllTypesExhaustedError, AllRegionsExhaustedError
  tests/
    test_param_separator.py
    test_transform_adapter.py
    test_sequential_engine.py
    test_race_engine.py
    test_intelligent_router.py
    test_launcher.py
    test_entry_point.py
    test_cross_region_manager.py
    test_prepare_regions.py
    test_quota_fetcher.py
    integration/
      test_real_aws.py          # Real AWS integration tests
```

## License

TBD

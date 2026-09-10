#!/usr/bin/env python3
"""SDK approach — race mode with multiple instance types.

Same model, same input data. Only differences:
1. Import launch_create_transform_job instead of boto3
2. InstanceType is a list instead of a string
3. Strategy="race" (or "sequential", or omit for AI expansion)
"""
import os
import sys
import time

# ── Configuration ──────────────────────────────────────────────
REGION = os.environ.get("SM_TEST_REGION", "us-east-1")
MODEL_NAME = os.environ.get("SM_TEST_MODEL_NAME", "")
S3_BUCKET = os.environ.get("SM_TEST_S3_BUCKET", "")

if not MODEL_NAME or not S3_BUCKET:
    print("Set SM_TEST_MODEL_NAME and SM_TEST_S3_BUCKET env vars first.")
    sys.exit(1)


def run_sdk(job_name_prefix: str, input_uri: str, output_prefix: str):
    """Run the SDK approach with race mode."""
    # This is the ONLY import change from standard boto3
    from sm_resilient_launcher import launch_create_transform_job
    
    events = []
    events.append((time.time(), "Launching with race mode (3 types in parallel)..."))
    
    start = time.time()
    
    result = launch_create_transform_job(
        TransformJobName=job_name_prefix,
        ModelName=MODEL_NAME,
        TransformInput={
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": input_uri,
                }
            },
            "ContentType": "text/csv",
            "SplitType": "Line",
        },
        TransformOutput={
            "S3OutputPath": f"s3://{S3_BUCKET}/{output_prefix}/",
        },
        TransformResources={
            # THE KEY DIFFERENCE: a list of types instead of a single string
            "InstanceType": ["ml.g4dn.xlarge", "ml.g5.xlarge", "ml.g4dn.2xlarge"],
            "InstanceCount": 1,
        },
        # Strategy options (uncomment one):
        Strategy="race",          # All types in parallel — fastest wins
        # Strategy="sequential",  # Try types in order — most conservative
        # (omit Strategy for AI expansion — SDK picks the best approach)
    )
    
    elapsed = time.time() - start
    events.append((time.time(), f"✅ Winner: {result.instance_type_used}"))
    events.append((time.time(), f"   Region: {result.region_used}"))
    events.append((time.time(), f"   Job: {result.job_name}"))
    events.append((time.time(), f"   Strategy: {result.strategy_used}"))
    events.append((time.time(), f"   Attempts: {result.attempts}"))
    
    if result.errors:
        for e in result.errors:
            events.append((time.time(), f"   ⚠️  {e.instance_type}: {e.category.value}"))
    
    events.append((time.time(), f"   Total time: {elapsed:.1f}s"))
    
    return events, result.instance_type_used, elapsed


if __name__ == "__main__":
    print("Running SDK race mode approach...")
    events, winner, elapsed = run_sdk(
        "demo-sdk",
        f"s3://{S3_BUCKET}/demo/input/test.csv",
        "demo/sdk-output",
    )
    for ts, msg in events:
        print(f"  {msg}")
    print(f"\nResult: {winner} in {elapsed:.1f}s")

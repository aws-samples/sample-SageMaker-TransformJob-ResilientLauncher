#!/usr/bin/env python3
"""Standard boto3 approach — single instance type, no fallback.

This is what customers do today. When the instance type has no capacity
or quota, the job fails and they manually retry with a different type.
"""
import os
import sys
import time
import boto3

# ── Configuration ──────────────────────────────────────────────
REGION = os.environ.get("SM_TEST_REGION", "us-east-1")
MODEL_NAME = os.environ.get("SM_TEST_MODEL_NAME", "")
S3_BUCKET = os.environ.get("SM_TEST_S3_BUCKET", "")

if not MODEL_NAME or not S3_BUCKET:
    print("Set SM_TEST_MODEL_NAME and SM_TEST_S3_BUCKET env vars first.")
    sys.exit(1)


def run_standard(job_name_prefix: str, input_uri: str, output_prefix: str):
    """Run the standard boto3 approach with manual retries."""
    sm = boto3.client("sagemaker", region_name=REGION)
    
    # The customer's retry list — they have to do this manually
    types_to_try = [
        "ml.g5.2xlarge",     # First choice — but quota=0!
        # "ml.g5.4xlarge",   # Second choice — uncomment to retry manually
        # "ml.g4dn.xlarge",  # Third choice — uncomment to retry manually
    ]
    
    events = []
    
    for i, instance_type in enumerate(types_to_try):
        attempt = i + 1
        job_name = f"{job_name_prefix}-attempt-{attempt}"
        
        events.append((time.time(), f"Attempt {attempt}: Submitting {instance_type}..."))
        
        try:
            sm.create_transform_job(
                TransformJobName=job_name,
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
                    "InstanceType": instance_type,
                    "InstanceCount": 1,
                },
            )
            events.append((time.time(), f"Job created. Polling..."))
            
            # Poll until terminal
            while True:
                desc = sm.describe_transform_job(TransformJobName=job_name)
                status = desc["TransformJobStatus"]
                if status == "Completed":
                    events.append((time.time(), f"✅ SUCCESS on {instance_type}!"))
                    return events, instance_type, attempt
                elif status in ("Failed", "Stopped"):
                    reason = desc.get("FailureReason", "unknown")
                    events.append((time.time(), f"❌ FAILED: {reason[:60]}"))
                    break
                time.sleep(10)
                
        except Exception as e:
            error_msg = str(e)[:80]
            events.append((time.time(), f"❌ ERROR: {error_msg}"))
            
            if attempt < len(types_to_try):
                events.append((time.time(), "⏳ Customer must manually change type and retry..."))
                time.sleep(2)  # Simulate customer thinking/editing
            else:
                events.append((time.time(), "💀 No more types to try. Giving up."))
    
    return events, None, len(types_to_try)


if __name__ == "__main__":
    print("Running standard boto3 approach...")
    events, winner, attempts = run_standard(
        "demo-standard",
        f"s3://{S3_BUCKET}/demo/input/test.csv",
        "demo/standard-output",
    )
    for ts, msg in events:
        print(f"  {msg}")
    if winner:
        print(f"\nResult: Succeeded on {winner} after {attempts} attempt(s)")
    else:
        print(f"\nResult: FAILED after {attempts} attempt(s)")

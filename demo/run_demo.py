#!/usr/bin/env python3
"""Side-by-side demo: Standard boto3 vs SDK with capacity resilience.

Runs both approaches concurrently and shows a live split-screen comparison.

Usage:
    export SM_TEST_MODEL_NAME=my-model
    export SM_TEST_S3_BUCKET=sagemaker-us-east-1-123456789012
    
    python demo/run_demo.py

Estimated cost: ~$2 (GPU jobs run 1-2 minutes).
"""
import os
import sys
import time
import threading
import uuid
import boto3

# Ensure the SDK is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REGION = os.environ.get("SM_TEST_REGION", "us-east-1")
MODEL_NAME = os.environ.get("SM_TEST_MODEL_NAME", "")
S3_BUCKET = os.environ.get("SM_TEST_S3_BUCKET", "")

if not MODEL_NAME or not S3_BUCKET:
    print("ERROR: Set SM_TEST_MODEL_NAME and SM_TEST_S3_BUCKET env vars.")
    print()
    print("  export SM_TEST_MODEL_NAME=your-model-name")
    print("  export SM_TEST_S3_BUCKET=your-sagemaker-bucket")
    sys.exit(1)

# Test CSV
TEST_CSV = (
    "1.5,2.3,0.4,3.1,0.7,1.2,2.8\n"
    "0.2,1.1,0.8,2.0,0.3,0.9,1.5\n"
    "3.0,2.8,1.2,4.5,1.0,2.1,3.3\n"
)

W = 40  # Column width


def _pad(text: str, width: int = W) -> str:
    """Pad/truncate text to fit column width."""
    text = text[:width]
    return text + " " * (width - len(text))


def _header():
    print()
    print("╔" + "═" * W + "╦" + "═" * W + "╗")
    print("║" + _pad("  STANDARD boto3 (single type)") + "║" + _pad("  SDK (race mode, 3 types)") + "║")
    print("╠" + "═" * W + "╬" + "═" * W + "╣")


def _row(left: str, right: str):
    print("║" + _pad(left) + "║" + _pad(right) + "║")


def _footer():
    print("╚" + "═" * W + "╩" + "═" * W + "╝")


def _separator():
    print("╠" + "═" * W + "╬" + "═" * W + "╣")


def upload_test_data() -> str:
    """Upload test CSV and return the S3 URI."""
    s3 = boto3.client("s3", region_name=REGION)
    key = f"demo/{uuid.uuid4().hex[:8]}/input/test.csv"
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=TEST_CSV.encode())
    return f"s3://{S3_BUCKET}/{key}"


def run_standard(input_uri: str, output_prefix: str, events: list, lock: threading.Lock):
    """LEFT SIDE: Standard boto3 with manual retry."""
    sm = boto3.client("sagemaker", region_name=REGION)
    start = time.time()
    
    types_to_try = [
        ("ml.g5.2xlarge", "quota=0, will fail"),
        ("ml.g5.4xlarge", "quota=1, might work"),
    ]
    
    winner = None
    for i, (instance_type, note) in enumerate(types_to_try):
        elapsed = int(time.time() - start)
        job_name = f"demo-std-{uuid.uuid4().hex[:6]}"
        
        with lock:
            events.append(("left", f"[{elapsed:>3}s] Try #{i+1}: {instance_type}"))
            events.append(("left", f"       ({note})"))
        
        try:
            sm.create_transform_job(
                TransformJobName=job_name,
                ModelName=MODEL_NAME,
                TransformInput={
                    "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": input_uri}},
                    "ContentType": "text/csv",
                    "SplitType": "Line",
                },
                TransformOutput={"S3OutputPath": f"s3://{S3_BUCKET}/{output_prefix}-{i}/"},
                TransformResources={"InstanceType": instance_type, "InstanceCount": 1},
            )
            
            while True:
                desc = sm.describe_transform_job(TransformJobName=job_name)
                status = desc["TransformJobStatus"]
                elapsed = int(time.time() - start)
                
                if status == "Completed":
                    with lock:
                        events.append(("left", f"[{elapsed:>3}s] ✅ {instance_type} completed!"))
                    winner = instance_type
                    break
                elif status in ("Failed", "Stopped"):
                    reason = desc.get("FailureReason", "")[:50]
                    with lock:
                        events.append(("left", f"[{elapsed:>3}s] ❌ Failed: {reason}"))
                    if i < len(types_to_try) - 1:
                        with lock:
                            events.append(("left", f"[{elapsed:>3}s] ⏳ Must manually retry..."))
                    break
                
                with lock:
                    events.append(("left", f"[{elapsed:>3}s] ⏳ {status}..."))
                time.sleep(15)
            
            if winner:
                break
                
        except Exception as e:
            elapsed = int(time.time() - start)
            msg = str(e)[:50]
            with lock:
                events.append(("left", f"[{elapsed:>3}s] ❌ {msg}"))
            if i < len(types_to_try) - 1:
                with lock:
                    events.append(("left", f"[{elapsed:>3}s] ⏳ Must manually retry..."))
                time.sleep(2)
    
    elapsed = int(time.time() - start)
    if winner:
        with lock:
            events.append(("left", f"[{elapsed:>3}s] Done: {winner} in {elapsed}s"))
            events.append(("left_result", f"{winner} | {elapsed}s | {len(types_to_try)} tries"))
    else:
        with lock:
            events.append(("left", f"[{elapsed:>3}s] 💀 All attempts failed"))
            events.append(("left_result", f"FAILED | {elapsed}s | {len(types_to_try)} tries"))
    
    # Cleanup
    for i in range(len(types_to_try)):
        try:
            sm.stop_transform_job(TransformJobName=f"demo-std-{uuid.uuid4().hex[:6]}")
        except:
            pass


def run_sdk(input_uri: str, output_prefix: str, events: list, lock: threading.Lock):
    """RIGHT SIDE: SDK with race mode."""
    from sm_resilient_launcher import launch_create_transform_job, AllTypesExhaustedError
    
    start = time.time()
    
    with lock:
        events.append(("right", f"[  0s] Racing 3 types in parallel..."))
        events.append(("right", f"       → ml.g4dn.xlarge"))
        events.append(("right", f"       → ml.g5.xlarge"))
        events.append(("right", f"       → ml.g4dn.2xlarge"))
    
    try:
        result = launch_create_transform_job(
            TransformJobName=f"demo-sdk-{uuid.uuid4().hex[:6]}",
            ModelName=MODEL_NAME,
            TransformInput={
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": input_uri}},
                "ContentType": "text/csv",
                "SplitType": "Line",
            },
            TransformOutput={"S3OutputPath": f"s3://{S3_BUCKET}/{output_prefix}/"},
            TransformResources={
                "InstanceType": ["ml.g4dn.xlarge", "ml.g5.xlarge", "ml.g4dn.2xlarge"],
                "InstanceCount": 1,
            },
            Strategy="race",
        )
        
        elapsed = int(time.time() - start)
        with lock:
            events.append(("right", f"[{elapsed:>3}s] ✅ Winner: {result.instance_type_used}"))
            events.append(("right", f"[{elapsed:>3}s]    Job: {result.job_name[:30]}"))
            if result.errors:
                for e in result.errors:
                    events.append(("right", f"       ⚠️ {e.instance_type}: {e.category.value}"))
            events.append(("right", f"[{elapsed:>3}s] Done in {elapsed}s, automatic!"))
            events.append(("right_result", f"{result.instance_type_used} | {elapsed}s | automatic"))
    
    except AllTypesExhaustedError as e:
        elapsed = int(time.time() - start)
        with lock:
            events.append(("right", f"[{elapsed:>3}s] ❌ All types exhausted"))
            events.append(("right_result", f"FAILED | {elapsed}s | {len(e.errors)} errors"))
    
    except Exception as e:
        elapsed = int(time.time() - start)
        with lock:
            events.append(("right", f"[{elapsed:>3}s] ❌ Error: {str(e)[:40]}"))
            events.append(("right_result", f"ERROR | {elapsed}s"))


def main():
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  SageMaker Batch Transform: Capacity Resilience Demo │")
    print("  │  Standard boto3  vs  sm-resilient-launcher SDK       │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print(f"  Region:  {REGION}")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Bucket:  {S3_BUCKET}")
    print()
    
    # Upload test data
    print("  Uploading test data...", end=" ", flush=True)
    input_uri = upload_test_data()
    print("done.")
    print()
    print("  Starting both approaches concurrently...")
    print()
    
    events = []
    lock = threading.Lock()
    
    uid = uuid.uuid4().hex[:6]
    
    t_std = threading.Thread(
        target=run_standard,
        args=(input_uri, f"demo/std-{uid}", events, lock),
    )
    t_sdk = threading.Thread(
        target=run_sdk,
        args=(input_uri, f"demo/sdk-{uid}", events, lock),
    )
    
    t_std.start()
    t_sdk.start()
    
    # Live display
    left_lines = []
    right_lines = []
    left_result = ""
    right_result = ""
    displayed = 0
    
    _header()
    
    while t_std.is_alive() or t_sdk.is_alive() or len(events) > displayed:
        with lock:
            while displayed < len(events):
                side, msg = events[displayed]
                if side == "left":
                    left_lines.append(msg)
                    _row(msg, "")
                elif side == "right":
                    right_lines.append(msg)
                    _row("", msg)
                elif side == "left_result":
                    left_result = msg
                elif side == "right_result":
                    right_result = msg
                displayed += 1
        time.sleep(0.5)
    
    # Final summary
    _separator()
    _row(f"  Result: {left_result}", f"  Result: {right_result}")
    _footer()
    
    # Cleanup
    print()
    print("  Cleaning up test data...", end=" ", flush=True)
    s3 = boto3.client("s3", region_name=REGION)
    prefix = input_uri.replace(f"s3://{S3_BUCKET}/", "").rsplit("/", 1)[0]
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"demo/"):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket=S3_BUCKET, Key=obj["Key"])
    print("done.")
    print()


if __name__ == "__main__":
    main()

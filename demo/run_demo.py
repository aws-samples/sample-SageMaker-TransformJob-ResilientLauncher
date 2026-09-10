#!/usr/bin/env python3
"""Side-by-side demo: Standard boto3 vs SDK with capacity resilience.

Phase 1: Runs both approaches concurrently and shows a live split-screen
         comparison with inference output and a summary table.
Phase 2: Demonstrates AI-powered instance selection via Bedrock.  A single
         scalar InstanceType plus a WorkloadDescription triggers the
         IntelligentRouter, which expands the type list, recommends a
         strategy, and explains its reasoning.

Usage:
    export SM_TEST_MODEL_NAME=my-model
    export SM_TEST_S3_BUCKET=sagemaker-us-east-1-123456789012

    python demo/run_demo.py

Estimated cost: ~$2 (GPU jobs run 1-2 minutes).
"""

import os
import sys
import threading
import time
import uuid

import boto3

# Ensure the SDK is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Config ─────────────────────────────────────────────────────

REGION = os.environ.get("SM_TEST_REGION", "us-east-1")
MODEL_NAME = os.environ.get("SM_TEST_MODEL_NAME", "")
S3_BUCKET = os.environ.get("SM_TEST_S3_BUCKET", "")

if not MODEL_NAME or not S3_BUCKET:
    print("ERROR: Set SM_TEST_MODEL_NAME and SM_TEST_S3_BUCKET env vars.")
    print()
    print("  export SM_TEST_MODEL_NAME=your-model-name")
    print("  export SM_TEST_S3_BUCKET=your-sagemaker-bucket")
    sys.exit(1)

# Instance types for the demo
STANDARD_TYPES = ["ml.g5.4xlarge", "ml.g5.2xlarge"]  # both quota=0, will fail
SDK_TYPES = ["ml.g5.4xlarge", "ml.g5.2xlarge", "ml.g4dn.xlarge"]  # includes fallback

STANDARD_NOTES = {
    "ml.g5.4xlarge": "preferred, but quota=0",
    "ml.g5.2xlarge": "backup, but quota=0",
}
SDK_NOTES = {
    "ml.g5.4xlarge": "preferred, but quota=0",
    "ml.g5.2xlarge": "backup, but quota=0",
    "ml.g4dn.xlarge": "available, quota=1",
}

# Test CSV
TEST_CSV = (
    "1.5,2.3,0.4,3.1,0.7,1.2,2.8\n"
    "0.2,1.1,0.8,2.0,0.3,0.9,1.5\n"
    "3.0,2.8,1.2,4.5,1.0,2.1,3.3\n"
)

W = 42  # Column width for the split-screen display

# ── Display helpers ────────────────────────────────────────────


def _pad(text: str, width: int = W) -> str:
    """Pad/truncate text to fit column width."""
    text = text[:width]
    return text + " " * (width - len(text))


def _header():
    print()
    print("\u2554" + "\u2550" * W + "\u2566" + "\u2550" * W + "\u2557")
    print(
        "\u2551"
        + _pad("  STANDARD boto3 (serial retries)")
        + "\u2551"
        + _pad("  SDK (race mode, all in parallel)")
        + "\u2551"
    )
    print("\u2560" + "\u2550" * W + "\u256c" + "\u2550" * W + "\u2563")


def _row(left: str, right: str):
    print("\u2551" + _pad(left) + "\u2551" + _pad(right) + "\u2551")


def _footer():
    print("\u255a" + "\u2550" * W + "\u2569" + "\u2550" * W + "\u255d")


def _separator():
    print("\u2560" + "\u2550" * W + "\u256c" + "\u2550" * W + "\u2563")


# ── S3 helpers ─────────────────────────────────────────────────


def upload_test_data() -> str:
    """Upload test CSV and return the S3 URI."""
    s3 = boto3.client("s3", region_name=REGION)
    key = f"demo/{uuid.uuid4().hex[:8]}/input/test.csv"
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=TEST_CSV.encode())
    return f"s3://{S3_BUCKET}/{key}"


def download_output(output_s3_prefix: str) -> str:
    """Download the first .out file under a prefix and return its contents."""
    s3 = boto3.client("s3", region_name=REGION)
    if output_s3_prefix.startswith("s3://"):
        without_scheme = output_s3_prefix[len("s3://"):]
        bucket = without_scheme.split("/", 1)[0]
        prefix = without_scheme.split("/", 1)[1] if "/" in without_scheme else ""
    else:
        bucket = S3_BUCKET
        prefix = output_s3_prefix

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".out"):
                resp = s3.get_object(Bucket=bucket, Key=obj["Key"])
                return resp["Body"].read().decode("utf-8").strip()
    return "(no output found)"


def cleanup_s3():
    """Remove all objects under the demo/ prefix."""
    s3 = boto3.client("s3", region_name=REGION)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="demo/"):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket=S3_BUCKET, Key=obj["Key"])


# ── Phase 1: Standard boto3 side (LEFT) ───────────────────────


def run_standard(
    input_uri: str,
    output_prefix: str,
    events: list,
    lock: threading.Lock,
    result_store: dict,
    standard_done: threading.Event | None = None,
):
    """LEFT SIDE: Standard boto3 with serial manual retries.

    Tries g5.4xlarge then g5.2xlarge (both quota=0) with a simulated
    5-second manual delay between attempts.  Gets stuck after two
    failures.
    """
    sm = boto3.client("sagemaker", region_name=REGION)
    start = time.time()

    winner = None
    winner_output_path = None
    total_attempts = 0
    output_path = f"s3://{S3_BUCKET}/{output_prefix}/"

    for i, instance_type in enumerate(STANDARD_TYPES):
        total_attempts += 1
        elapsed = int(time.time() - start)
        job_name = f"demo-std-{uuid.uuid4().hex[:6]}"
        note = STANDARD_NOTES.get(instance_type, "")

        with lock:
            events.append(("left", f"[{elapsed:>3}s] Try #{i+1}: {instance_type}"))
            events.append(("left", f"       ({note})"))

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
                TransformOutput={"S3OutputPath": output_path},
                TransformResources={
                    "InstanceType": instance_type,
                    "InstanceCount": 1,
                },
            )

            while True:
                desc = sm.describe_transform_job(TransformJobName=job_name)
                status = desc["TransformJobStatus"]
                elapsed = int(time.time() - start)

                if status == "Completed":
                    with lock:
                        events.append(
                            ("left", f"[{elapsed:>3}s] \u2705 {instance_type} completed!")
                        )
                    winner = instance_type
                    winner_output_path = output_path
                    break
                elif status in ("Failed", "Stopped"):
                    reason = desc.get("FailureReason", "")[:45]
                    with lock:
                        events.append(
                            ("left", f"[{elapsed:>3}s] \u274c Failed: {reason}")
                        )
                    break

                with lock:
                    events.append(("left", f"[{elapsed:>3}s] \u23f3 {status}..."))
                time.sleep(15)

            if winner:
                break

            # Simulate manual retry delay
            elapsed = int(time.time() - start)
            with lock:
                events.append(("left", f"[{elapsed:>3}s] \u23f3 Manual retry delay (5s)..."))
                events.append(("left", "       Customer editing code..."))
            time.sleep(5)

        except Exception as e:
            elapsed = int(time.time() - start)
            msg = str(e)[:45]
            with lock:
                events.append(("left", f"[{elapsed:>3}s] \u274c {msg}"))

            # Simulate manual retry delay before next attempt
            if i < len(STANDARD_TYPES) - 1:
                elapsed = int(time.time() - start)
                with lock:
                    events.append(("left", f"[{elapsed:>3}s] \u23f3 Manual retry delay (5s)..."))
                    events.append(("left", "       Customer editing code..."))
                time.sleep(5)

    elapsed = int(time.time() - start)
    if winner:
        with lock:
            events.append(("left", f"[{elapsed:>3}s] Done: {winner} in {elapsed}s"))
            events.append(("left_result", f"{winner} | {elapsed}s | {total_attempts} tries"))
        result_store["left"] = {
            "winner": winner,
            "elapsed": elapsed,
            "attempts": total_attempts,
            "output_path": winner_output_path,
            "manual_interventions": total_attempts - 1,
        }
    else:
        with lock:
            events.append(("left", f"[{elapsed:>3}s] \U0001f480 All attempts failed"))
            events.append(("left_result", f"FAILED | {elapsed}s | {total_attempts} tries"))
        result_store["left"] = {
            "winner": None,
            "elapsed": elapsed,
            "attempts": total_attempts,
            "output_path": None,
            "manual_interventions": total_attempts - 1,
        }

    # Signal that the standard side is done so SDK can start
    if standard_done is not None:
        standard_done.set()


# ── Phase 1: SDK side (RIGHT) ─────────────────────────────────


def run_sdk(
    input_uri: str,
    output_prefix: str,
    events: list,
    lock: threading.Lock,
    result_store: dict,
    standard_done: threading.Event | None = None,
):
    """RIGHT SIDE: SDK with race mode — with live log capture."""
    from sm_resilient_launcher import launch_create_transform_job, AllTypesExhaustedError
    import logging

    if standard_done is not None:
        standard_done.wait()

    start = time.time()

    with lock:
        events.append(("right", f"[  0s] Racing {len(SDK_TYPES)} types in parallel..."))
        for itype in SDK_TYPES:
            note = SDK_NOTES.get(itype, "")
            events.append(("right", f"       \u2192 {itype} ({note})"))

    # Set up a log handler that captures SDK log messages
    class DemoLogHandler(logging.Handler):
        def emit(self, record):
            elapsed = int(time.time() - start)
            msg = record.getMessage()
            # Filter for interesting messages
            if any(kw in msg.lower() for kw in [
                "failed", "cancel", "winner", "skipping", "moving",
                "exhausted", "capacity", "quota",
            ]):
                short_msg = msg[:38]
                with lock:
                    events.append(("right", f"[{elapsed:>3}s] {short_msg}"))

    handler = DemoLogHandler()
    handler.setLevel(logging.INFO)

    # Attach to SDK loggers
    sdk_loggers = [
        logging.getLogger("sm_resilient_launcher._race_engine"),
        logging.getLogger("sm_resilient_launcher._sequential_engine"),
        logging.getLogger("sm_resilient_launcher._error_classifier"),
    ]
    for lg in sdk_loggers:
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)

    try:
        output_path = f"s3://{S3_BUCKET}/{output_prefix}/"
        result = launch_create_transform_job(
            TransformJobName=f"demo-sdk-{uuid.uuid4().hex[:6]}",
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
            TransformOutput={"S3OutputPath": output_path},
            TransformResources={
                "InstanceType": SDK_TYPES,
                "InstanceCount": 1,
            },
            Strategy="race",
        )

        elapsed = int(time.time() - start)
        with lock:
            events.append(("right", f"[{elapsed:>3}s] \u2705 Winner: {result.instance_type_used}"))
            events.append(("right", f"[{elapsed:>3}s]    Job: {result.job_name[:35]}"))
            if result.errors:
                for e in result.errors:
                    events.append(
                        ("right", f"       \u26a0\ufe0f  {e.instance_type}: {e.category.value}")
                    )
            events.append(("right", f"[{elapsed:>3}s] Done in {elapsed}s, automatic!"))
            events.append(("right_result", f"{result.instance_type_used} | {elapsed}s | automatic"))
        result_store["right"] = {
            "winner": result.instance_type_used,
            "elapsed": elapsed,
            "attempts": result.attempts,
            "output_path": output_path,
            "manual_interventions": 0,
            "errors": [f"{e.instance_type}: {e.category.value}" for e in result.errors],
        }

    except AllTypesExhaustedError as e:
        elapsed = int(time.time() - start)
        with lock:
            events.append(("right", f"[{elapsed:>3}s] \u274c All types exhausted"))
            events.append(("right_result", f"FAILED | {elapsed}s | {len(e.errors)} errors"))
        result_store["right"] = {
            "winner": None,
            "elapsed": elapsed,
            "attempts": len(e.errors),
            "output_path": None,
            "manual_interventions": 0,
        }

    except Exception as e:
        elapsed = int(time.time() - start)
        with lock:
            events.append(("right", f"[{elapsed:>3}s] \u274c Error: {str(e)[:40]}"))
            events.append(("right_result", f"ERROR | {elapsed}s"))
        result_store["right"] = {
            "winner": None,
            "elapsed": elapsed,
            "attempts": 0,
            "output_path": None,
            "manual_interventions": 0,
        }

    finally:
        # Remove the handler to avoid side effects
        for lg in sdk_loggers:
            lg.removeHandler(handler)


# ── Phase 1: Output comparison ────────────────────────────────


def show_inference_output(result_store: dict):
    """Download and display inference predictions from both sides."""
    print()
    print("  \u250c" + "\u2500" * 48 + "\u2510")
    print("  \u2502  Inference Output Comparison" + " " * 19 + "\u2502")
    print("  \u2514" + "\u2500" * 48 + "\u2518")
    print()

    left = result_store.get("left", {})
    right = result_store.get("right", {})
    left_output = None
    right_output = None

    if left.get("output_path"):
        print("  Downloading standard boto3 output...", end=" ", flush=True)
        try:
            left_output = download_output(left["output_path"])
            print("done.")
        except Exception as e:
            print(f"failed: {e}")

    if right.get("output_path"):
        print("  Downloading SDK race mode output...", end=" ", flush=True)
        try:
            right_output = download_output(right["output_path"])
            print("done.")
        except Exception as e:
            print(f"failed: {e}")

    print()
    print("\u2554" + "\u2550" * W + "\u2566" + "\u2550" * W + "\u2557")
    print(
        "\u2551"
        + _pad("  Standard boto3 predictions")
        + "\u2551"
        + _pad("  SDK race mode predictions")
        + "\u2551"
    )
    print("\u2560" + "\u2550" * W + "\u256c" + "\u2550" * W + "\u2563")

    left_lines = (left_output or "(no output)").split("\n")
    right_lines = (right_output or "(no output)").split("\n")
    max_lines = max(len(left_lines), len(right_lines))

    for i in range(max_lines):
        l = left_lines[i] if i < len(left_lines) else ""
        r = right_lines[i] if i < len(right_lines) else ""
        _row(f"  {l}", f"  {r}")

    _separator()
    if left_output and right_output:
        if left_output.strip() == right_output.strip():
            _row("  \u2705 Outputs match!", "  \u2705 Outputs match!")
        else:
            _row("  \u26a0\ufe0f  Outputs differ", "  \u26a0\ufe0f  Outputs differ")
    else:
        if not left_output:
            _row("  (no output to compare)", "")
        if not right_output:
            _row("", "  (no output to compare)")
    _footer()


def show_summary(result_store: dict):
    """Print the final Phase 1 comparison summary."""
    left = result_store.get("left", {})
    right = result_store.get("right", {})

    print()
    print("  \u250c" + "\u2500" * 48 + "\u2510")
    print("  \u2502  Final Comparison Summary" + " " * 22 + "\u2502")
    print("  \u2514" + "\u2500" * 48 + "\u2518")
    print()

    rows = [
        ("Metric", "Standard boto3", "SDK race mode"),
        ("Total time", f"{left.get('elapsed', '?')}s", f"{right.get('elapsed', '?')}s"),
        ("Instance used", left.get("winner") or "FAILED", right.get("winner") or "FAILED"),
        ("Attempts", str(left.get("attempts", "?")), str(right.get("attempts", "?"))),
        (
            "Manual work",
            f"{left.get('manual_interventions', '?')} intervention(s)",
            "0 (automatic)",
        ),
        ("Approach", "Serial, with manual delay", "Parallel race, auto-fallback"),
    ]

    col_w = [18, 28, 28]
    sep = "  +" + "+".join("-" * c for c in col_w) + "+"

    print(sep)
    for i, (c0, c1, c2) in enumerate(rows):
        print(
            "  |"
            + f" {c0}".ljust(col_w[0])
            + "|"
            + f" {c1}".ljust(col_w[1])
            + "|"
            + f" {c2}".ljust(col_w[2])
            + "|"
        )
        if i == 0:
            print(sep)
    print(sep)

    left_t = left.get("elapsed", 0)
    right_t = right.get("elapsed", 0)
    if left_t and right_t and left_t > right_t:
        saved = left_t - right_t
        print()
        print(f"  \u23e9 SDK was {saved}s faster ({saved // 60}m {saved % 60}s saved)")
    elif right_t and left_t and right_t > left_t:
        print()
        print(f"  (Standard was {right_t - left_t}s faster this run)")

    print()
    if left.get("winner") and right.get("winner"):
        print("  \u2705 Same model, same data, same results \u2014 SDK just gets there faster")
        print("     with zero manual intervention.")
    print()


# ── Phase 2: AI-Powered Instance Selection ─────────────────────


def run_ai_expansion(input_uri: str):
    """Phase 2: demonstrate Bedrock-powered intelligent routing.

    Calls IntelligentRouter.generate_plan() directly first so the user
    sees the AI reasoning, expanded type list, and recommended strategy
    immediately — before the (potentially long) job execution starts.
    Then passes the AI-recommended types and strategy to the SDK so it
    skips the second Bedrock call and goes straight to job execution.
    """
    from sm_resilient_launcher import launch_create_transform_job, AllTypesExhaustedError
    from sm_resilient_launcher._quota_fetcher import QuotaFetcher
    from sm_resilient_launcher._pricing_fetcher import PricingFetcher
    from sm_resilient_launcher._type_expander import TypeExpander
    from sm_resilient_launcher._intelligent_router import IntelligentRouter

    box_w = 66
    print()
    print("  \u250c" + "\u2500" * box_w + "\u2510")
    print("  \u2502" + "  Phase 2: AI-Powered Instance Selection (Bedrock)".ljust(box_w) + "\u2502")
    print("  \u2514" + "\u2500" * box_w + "\u2518")
    print()

    scalar_type = "ml.g4dn.xlarge"
    workload_desc = (
        "XGBoost binary classification, ~200MB model, "
        "cost-sensitive batch inference"
    )

    print(f'  Input: InstanceType="{scalar_type}" (scalar, no strategy specified)')
    print(f'         WorkloadDescription="{workload_desc}"')
    print()

    # ── Step 1: Get candidates deterministically ──

    quota_fetcher = QuotaFetcher(REGION)
    pricing_fetcher = PricingFetcher()
    type_expander = TypeExpander(quota_fetcher)
    router = IntelligentRouter(quota_fetcher, pricing_fetcher, type_expander, REGION)

    # Discover ALL transform types from the account's Service Quotas
    all_transform_types = quota_fetcher.get_all_transform_types()
    # Filter to GPU types (what the IntelligentRouter sends to Bedrock)
    gpu_prefixes = ("ml.g", "ml.p", "ml.trn", "ml.inf")
    candidates = [t for t in all_transform_types if any(t.startswith(p) for p in gpu_prefixes)]
    # Ensure original type is included
    if scalar_type not in candidates:
        candidates.insert(0, scalar_type)
    print(f"  GPU types in account (from Service Quotas API): {len(candidates)} types found")
    for t in candidates:
        print(f"    {t}")
    print()

    # ── Step 2: Fetch quotas (deterministic) ──

    quotas = quota_fetcher.fetch_quotas(candidates)
    non_zero = {t: q for t, q in quotas.items() if q and q > 0}
    print(f"  Types with quota > 0: {len(non_zero)}")
    for t, q in sorted(non_zero.items()):
        print(f"    {t}: quota={q}")
    print()

    # ── Step 3: Build and show the prompt ──

    prompt = router._build_prompt(scalar_type, candidates, workload_desc, quotas)
    print("  Prompt sent to Bedrock:")
    print("  " + "\u2500" * 60)
    for line in prompt.split("\n"):
        print(f"  \u2502 {line}")
    print("  " + "\u2500" * 60)
    print()

    # ── Step 4: Call Bedrock ──

    print("  Calling Bedrock Converse API...", flush=True)
    bedrock_start = time.time()

    try:
        expanded_types, strategy, reasoning = router.generate_plan(
            scalar_type, workload_desc
        )
    except Exception as e:
        print(f"  \u274c Bedrock call failed: {e}")
        print()
        return

    bedrock_elapsed = time.time() - bedrock_start
    print(f"  \u2705 Bedrock responded! ({bedrock_elapsed:.1f}s)")
    print()

    # Show AI reasoning immediately
    if reasoning:
        print("  AI Reasoning:")
        for line in reasoning.split("\n"):
            stripped = line.strip()
            if stripped:
                print(f"    {stripped}")
        print()

    print(f"  Expanded types: {expanded_types}")
    print(f"  Recommended strategy: {strategy}")
    print()

    # ── Step 2: Execute with the AI-recommended plan ───────────────────

    output_path = f"s3://{S3_BUCKET}/demo/ai-{uuid.uuid4().hex[:6]}/"
    job_start = time.time()

    print("  Executing job with AI-recommended plan...")
    print("  (provisioning + inference typically takes 1-3 minutes)")
    print()

    try:
        result = launch_create_transform_job(
            TransformJobName=f"demo-ai-{uuid.uuid4().hex[:6]}",
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
            TransformOutput={"S3OutputPath": output_path},
            TransformResources={
                "InstanceType": expanded_types,
                "InstanceCount": 1,
            },
            Strategy=strategy,
        )

        job_elapsed = int(time.time() - job_start)
        total_elapsed = int(time.time() - bedrock_start)

        print(f"  Strategy used: {result.strategy_used}")
        print(f"  Instance used: {result.instance_type_used}")
        print()

        # Display errors from failed attempts (if any)
        if result.errors:
            print("  Fallback attempts:")
            for e in result.errors:
                print(f"    \u26a0\ufe0f  {e.instance_type}: {e.category.value}")
            print()

        print(f"  [{total_elapsed:>3}s] \u2705 Completed on {result.instance_type_used} "
              f"(Bedrock: {bedrock_elapsed:.1f}s, job: {job_elapsed}s)")

        # Download and show predictions
        try:
            ai_output = download_output(output_path)
            print()
            print(f"  Predictions: {ai_output}")
        except Exception:
            print()
            print("  Predictions: (download failed)")

    except AllTypesExhaustedError as e:
        total_elapsed = int(time.time() - bedrock_start)
        print(f"  [{total_elapsed:>3}s] \u274c All types exhausted")
        for err in e.errors:
            print(f"    \u26a0\ufe0f  {err.instance_type}: {err.category.value}")

    except Exception as e:
        total_elapsed = int(time.time() - bedrock_start)
        print(f"  [{total_elapsed:>3}s] \u274c Error: {e}")

    print()


# ── Main ───────────────────────────────────────────────────────


def main():
    types_display = ", ".join(SDK_TYPES)

    print()
    print("  \u250c" + "\u2500" * 56 + "\u2510")
    print("  \u2502  SageMaker Batch Transform: Capacity Resilience Demo  \u2502")
    print("  \u2502  Standard boto3  vs  sm-resilient-launcher SDK        \u2502")
    print("  \u2514" + "\u2500" * 56 + "\u2518")
    print()
    print(f"  Region:      {REGION}")
    print(f"  Model:       {MODEL_NAME}")
    print(f"  Bucket:      {S3_BUCKET}")
    print(f"  Types:       {types_display}")
    print()
    print("  LEFT  (standard): g5.4xlarge \u2192 fail \u2192 manual retry \u2192 g5.2xlarge \u2192 fail \u2192 STUCK")
    print("  RIGHT (SDK):      all 3 raced, g4dn.xlarge wins automatically")
    print()

    # Upload test data
    print("  Uploading test data...", end=" ", flush=True)
    input_uri = upload_test_data()
    print("done.")
    print()

    # ── Phase 1: Side-by-side comparison ───────────────────────

    print("  Starting both approaches (standard first, then SDK)...")

    events = []
    lock = threading.Lock()
    result_store = {}
    standard_done = threading.Event()

    uid = uuid.uuid4().hex[:6]

    t_std = threading.Thread(
        target=run_standard,
        args=(input_uri, f"demo/std-{uid}", events, lock, result_store, standard_done),
    )
    t_sdk = threading.Thread(
        target=run_sdk,
        args=(input_uri, f"demo/sdk-{uid}", events, lock, result_store, standard_done),
    )

    t_std.start()
    t_sdk.start()

    # Live display
    left_result = ""
    right_result = ""
    displayed = 0

    _header()

    while t_std.is_alive() or t_sdk.is_alive() or len(events) > displayed:
        with lock:
            while displayed < len(events):
                side, msg = events[displayed]
                if side == "left":
                    _row(msg, "")
                elif side == "right":
                    _row("", msg)
                elif side == "left_result":
                    left_result = msg
                elif side == "right_result":
                    right_result = msg
                displayed += 1
        time.sleep(0.5)

    _separator()
    _row(f"  Result: {left_result}", f"  Result: {right_result}")
    _footer()

    # Inference output comparison
    show_inference_output(result_store)

    # Summary table
    show_summary(result_store)

    # ── Phase 2: AI Expansion ──────────────────────────────────

    run_ai_expansion(input_uri)

    # ── Cleanup ────────────────────────────────────────────────

    print("  Cleaning up test data...", end=" ", flush=True)
    cleanup_s3()
    print("done.")
    print()


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""
Quantization accuracy evaluation module.
"""

from __future__ import annotations

# Standard
import argparse
import asyncio
import json
import os
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Third Party
from openai import AsyncOpenAI
from tqdm import tqdm

# Constants
LOCAL_DATASETS_DIR = Path("/home/fei/research/datasets")
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DATASET_SPLITS = {
    "gsm8k": "test",
}
DATASET_PROMPT_FILES = {
    "gsm8k": "/home/fei/research/llm/KVCache/LMCache-Quant-CC/benchmarks/quantization/lib_prompts/gsm8k_prompt_original.txt",
}

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

@dataclass
class Sample:
    question: str
    prompt: str
    gold: str

@dataclass(frozen=True)
class SampleResult:
    question: str
    generation: str
    answer: str
    list_from_pred: List[str]
    list_from_answer: List[str]
    pred: Optional[float]
    label: float
    is_pred_true: bool

@dataclass(frozen=True)
class EvaluationResults:
    samples: List[SampleResult]
    accuracy: float

# ==============================================================================
# 1. Extraction & Parsing Logic
# ==============================================================================
def _evaluate_pred_answer(pred_str: str, ans_str: str) -> Tuple[bool, Optional[float], List[str], float, List[str]]:
    pattern = r"\d*\.?\d+"
    pred_str = pred_str.replace(",", "")
    ans_str = ans_str.replace(",", "")
    pred_list = re.findall(pattern, pred_str)
    gold_list = re.findall(pattern, ans_str)
    if len(gold_list) == 0:
        return False, None, pred_list, float("nan"), gold_list
    if len(pred_list) >= 1:
        pred = float(pred_list[-1])
        gold = float(gold_list[-1])
        is_pred_true = pred == gold
    else:
        is_pred_true = False
        pred = None
        gold = float(gold_list[-1])
    return is_pred_true, pred, pred_list, gold, gold_list


def _resolve_base_url(base_url: str, host: Optional[str], port: Optional[int]) -> str:
    if (host is not None or port is not None) and base_url == DEFAULT_BASE_URL:
        host = host or "localhost"
        port = port or 8000
        return f"http://{host}:{port}/v1"
    return base_url

# ==============================================================================
# 2. Dataset Loading & Building
# ==============================================================================

def _load_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line: items.append(json.loads(line))
    return items

def _load_local_data(name: str, split: str) -> List[Dict[str, Any]]:
    """Generic local dataset loader supporting parquet and jsonl."""
    # Try full name and base name
    candidates = [LOCAL_DATASETS_DIR / name, LOCAL_DATASETS_DIR / name.split("/")[-1]]
    local_path = next((c for c in candidates if c.exists()), None)

    if not local_path:
        raise FileNotFoundError(f"Dataset '{name}' not found in {LOCAL_DATASETS_DIR}")

    # Search for files
    files = list(local_path.rglob("*.parquet")) + list(local_path.rglob("*.jsonl"))
    # Filter by split
    split_files = [f for f in files if split in f.name] or files
    if not split_files:
        raise FileNotFoundError(f"No data files found for split '{split}' in {local_path}")
    
    target_file = split_files[0]
    if target_file.suffix == ".parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(target_file)
        return [dict(zip(table.column_names, row)) for row in zip(*table.to_pydict().values())]
    return _load_jsonl(target_file)

def _build_samples(data_name: str,
                   data: List[dict],
                   prompt_prefix: Optional[str],
                   zero_shot: bool) -> List[Sample]:
    """Convert raw data dicts to Sample objects.
    Use the `<think>` style prompt for all models when not zero-shot.
    """
    samples = []
    if data_name == "gsm8k":
        for i, item in enumerate(data):
            q = item.get("question") or item.get("prompt")
            a = item.get("answer") or item.get("gold")
            if not q or a is None:
                continue
            if zero_shot:
                prompt = "Answer the question through the form of The answer is xxx. Do not generate others."
                prompt = f"task {i} : {prompt}\nQuestion: {q}\n"
            else:
                assert prompt_prefix, "Prompt prefix must be provided for non-zero-shot evaluation."
                prompt = f"task {i} : {prompt_prefix}\nQuestion: {q} \n\n<think>\n\n</think>\n\n "
            samples.append(Sample(question=q, prompt=prompt, gold=str(a)))
    else:
        raise NotImplementedError(f"Dataset '{data_name}' not supported yet.")
    return samples

# ==============================================================================
# 3. Generation & Evaluation
# ==============================================================================

async def _api_call(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
) -> str:
    """Call the completions API only."""
    try:
        resp = await client.completions.create(model=model,
                                               prompt=prompt,
                                               max_tokens=max_tokens,
                                               temperature=0.0)
        return resp.choices[0].text or ""
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return ""

async def _evaluate_samples(samples: List[Sample],
                            base_url: str,
                            args: argparse.Namespace) -> EvaluationResults:
    client = AsyncOpenAI(base_url=base_url, api_key="dummy_key")
    sem = asyncio.Semaphore(args.concurrency)
    
    async def check(s: Sample) -> SampleResult:
        async with sem:
            out = await _api_call(client, args.model, s.prompt, args.max_new_tokens)
            is_pred_true, pred, pred_list, gold, gold_list = _evaluate_pred_answer(out, s.gold)
            return SampleResult(
                question=s.question,
                generation=out,
                answer=s.gold,
                list_from_pred=pred_list,
                list_from_answer=gold_list,
                pred=pred,
                label=gold,
                is_pred_true=is_pred_true,
            )

    tasks = [asyncio.create_task(check(s)) for s in samples]
    samples_out: List[SampleResult] = []
    with tqdm(total=len(tasks), desc="Evaluating", unit="samples") as pbar:
        for task in asyncio.as_completed(tasks):
            samples_out.append(await task)
            pbar.update(1)
    accuracy = sum(sample.is_pred_true for sample in samples_out) / len(samples_out) if samples_out else 0.0
    return EvaluationResults(samples=samples_out, accuracy=accuracy)

# ==============================================================================
# 4. Main Workflow Phases
# ==============================================================================

def accuracy_eval(args: argparse.Namespace) -> None:
    split = DATASET_SPLITS.get(args.dataset, "test")
    data = _load_local_data(args.dataset, split)

    prompt_prefix = ""
    if not args.zero_shot:
        prompt_path = DATASET_PROMPT_FILES.get(args.dataset)
        if not prompt_path:
            logger.error(f"No prompt file found for dataset '{args.dataset}'")
            return
        prompt_prefix = Path(prompt_path).read_text(encoding="utf-8")
    samples = _build_samples(args.dataset, data, prompt_prefix, args.zero_shot)
    
    # samples = samples[:100]  # Limit to first 100 samples for quick testing
    
    if not samples:
        logger.error("No samples loaded.")
        return
    logger.info(f"Loaded {len(samples)} samples for evaluation.")

    base_url = _resolve_base_url(args.base_url, args.host, args.port)

    logger.info("Phase 1: Baseline Evaluation...")
    b_res = asyncio.run(_evaluate_samples(samples, base_url, args))
    logger.info(f"Baseline accuracy: {b_res.accuracy:.4f}")
    
    logger.info("Phase 2: Quantized Evaluation...")
    q_res = asyncio.run(_evaluate_samples(samples, base_url, args))
    logger.info(f"Quantized accuracy: {q_res.accuracy:.4f}")
    
    exit()

    diff = q_res.accuracy - b_res.accuracy
    print("\n" + "="*30 + " SUMMARY " + "="*30)
    print(f"Baseline accuracy:  {b_res.accuracy:.4f}")
    print(f"Quantized accuracy: {q_res.accuracy:.4f}")
    print(f"Impact:    {diff:+.4f} ({diff*100:+.2f}%)")

    # Create output directory and file paths now, right before storing results
    model_name = Path(args.model).name
    output_dir = Path(__file__).parent / args.dataset / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_file = output_dir / "generation_results.txt"
    evaluation_result_file = output_dir / f"evaluation_{args.dataset}.json"

    with evaluation_result_file.open("w", encoding="utf-8") as handle:
        json.dump({
            "samples": [sample.__dict__ for sample in q_res.samples],
            "metrics": {"accuracy": q_res.accuracy},
        }, handle, ensure_ascii=False, indent=2)

    with generation_file.open("w", encoding="utf-8") as handle:
        for sample in q_res.samples:
            handle.write(
                f"Q: {sample.question}\nA_model:\n{sample.generation}\nA:\n{sample.answer}\n\n"
            )

# ==============================================================================
# 5. CLI Definition
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()

    # Model/Server
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)

    # Dataset
    parser.add_argument("--dataset", choices=["gsm8k"], default="gsm8k")

    # Inference
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=30)
    # Completions-only mode (chat removed)
    parser.add_argument("--root-output-dir", type=str, default="outputs")
    parser.add_argument("--zero-shot", action="store_true", default=False)

    args = parser.parse_args()
    accuracy_eval(args)

if __name__ == "__main__":
    main()

"""
Amazon ML Worker — QLoRA Smoke Test.

Optional Qwen + QLoRA smoke-test support.
Default: 4-bit NF4 quantization + PEFT/LoRA.
Tests model loading, forward pass, backward pass, tiny training, checkpoint.

NEVER automatically runs full training. Smoke test only.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logging import get_logger

log = get_logger(__name__)


def check_qlora_dependencies() -> Dict[str, bool]:
    """Check which QLoRA dependencies are available."""
    deps = {}
    for pkg in ["torch", "transformers", "peft", "bitsandbytes", "accelerate", "trl"]:
        try:
            __import__(pkg)
            deps[pkg] = True
        except ImportError:
            deps[pkg] = False
    return deps


def run_qlora_smoke_test(
    model_name: str = "Qwen/Qwen2-0.5B",
    bits: int = 4,
    max_steps: int = 5,
    max_seq_len: int = 128,
    output_dir: str = "outputs/qlora_smoke",
) -> Dict[str, Any]:
    """
    Run a QLoRA smoke test.

    Tests: model loading → quantization → LoRA → forward → backward → tiny train → checkpoint.

    Returns a report dict.
    """
    report: Dict[str, Any] = {"status": "not_started", "steps": {}}

    # 1. Check dependencies
    deps = check_qlora_dependencies()
    report["dependencies"] = deps

    missing = [k for k, v in deps.items() if not v]
    if missing:
        report["status"] = "skipped"
        report["reason"] = f"Missing dependencies: {missing}"
        log.warning("QLoRA smoke test SKIPPED: missing %s", missing)
        return report

    # 2. Check CUDA
    import torch
    if not torch.cuda.is_available():
        report["status"] = "skipped"
        report["reason"] = "CUDA not available"
        log.warning("QLoRA smoke test SKIPPED: no CUDA GPU.")
        return report

    try:
        report["gpu"] = torch.cuda.get_device_name(0)
        report["vram_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 2)
    except Exception:
        pass

    start = time.time()

    try:
        # 3. Load model with quantization
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType

        log.info("Loading model: %s (%d-bit)", model_name, bits)
        report["steps"]["load_start"] = True

        quant_config = BitsAndBytesConfig(
            load_in_4bit=(bits == 4),
            load_in_8bit=(bits == 8),
            bnb_4bit_quant_type="nf4" if bits == 4 else None,
            bnb_4bit_compute_dtype=torch.float16 if bits == 4 else None,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        report["steps"]["model_loaded"] = True
        log.info("Model loaded successfully.")

        # 4. Attach LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        report["trainable_params"] = trainable
        report["total_params"] = total
        report["trainable_pct"] = round(trainable / max(total, 1) * 100, 4)
        report["steps"]["lora_attached"] = True
        log.info("LoRA attached: %d trainable / %d total params (%.4f%%)", trainable, total, report["trainable_pct"])

        # 5. Forward pass
        test_text = "This is a smoke test for QLoRA fine-tuning."
        inputs = tokenizer(test_text, return_tensors="pt", max_length=max_seq_len, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        outputs = model(**inputs)
        report["steps"]["forward_pass"] = True
        report["initial_loss"] = float(outputs.loss.item())
        log.info("Forward pass OK. Loss: %.4f", report["initial_loss"])

        # 6. Backward pass
        outputs.loss.backward()
        report["steps"]["backward_pass"] = True
        log.info("Backward pass OK.")

        # 7. Tiny training loop
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=1e-4)

        model.train()
        for step in range(max_steps):
            optimizer.zero_grad()
            outputs = model(**inputs)
            outputs.loss.backward()
            optimizer.step()
            log.info("  Step %d/%d, loss=%.4f", step + 1, max_steps, outputs.loss.item())

        report["final_loss"] = float(outputs.loss.item())
        report["steps"]["training"] = True
        log.info("Tiny training OK. Final loss: %.4f", report["final_loss"])

        # 8. Save checkpoint
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_path / "checkpoint"))
        tokenizer.save_pretrained(str(out_path / "checkpoint"))
        report["steps"]["checkpoint_saved"] = True
        log.info("Checkpoint saved.")

        # 9. Inference test
        model.eval()
        with torch.no_grad():
            gen_inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
            gen_outputs = model.generate(**gen_inputs, max_new_tokens=20)
            gen_text = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
        report["steps"]["inference"] = True
        report["generated_text"] = gen_text[:200]
        log.info("Inference test OK.")

        report["status"] = "passed"

    except Exception as e:
        report["status"] = "failed"
        report["error"] = str(e)
        log.error("QLoRA smoke test FAILED: %s", e)

    report["total_time"] = round(time.time() - start, 2)

    # Save report
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "smoke_test_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report

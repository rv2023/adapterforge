"""M6 Piece 2 — export the DistilBERT student to ONNX for Triton's ONNX backend.

PyTorch model -> static ONNX graph (+ weights) -> drop into Triton's model repository.
The student is an encoder/classifier (one forward pass, fixed graph) so ONNX fits
perfectly — unlike the autoregressive LLM (see docs/m6-serving-concepts.md §11c).

    python -m pipelines.export_student_onnx

Then run Triton (CPU, local Docker):
    docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
      -v "$PWD/serving/triton/model_repository:/models" \
      nvcr.io/nvidia/tritonserver:24.08-py3 tritonserver --model-repository=/models
"""

from pathlib import Path

import torch
import onnxruntime as ort
from transformers import AutoModelForSequenceClassification, AutoTokenizer

STUDENT_DIR = "models/distilbert-student"
OUT_PATH = "serving/triton/model_repository/distilbert-student/1/model.onnx"
# names MUST match config.pbtxt input/output names
INPUT_NAMES = ["input_ids", "attention_mask"]
OUTPUT_NAMES = ["logits"]


class LogitsOnly(torch.nn.Module):
    """Thin wrapper so the exported graph returns a plain logits TENSOR.

    HF models return a SequenceClassifierOutput object; torch.onnx.export wants tensors.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export() -> None:
    """Load the student, trace it to ONNX with dynamic batch+seq axes, verify, write."""
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_DIR,
        attn_implementation="eager",
    )
    model.eval()
    wrapped = LogitsOnly(model)
    dummy_batch = tokenizer(
        ["some sample text", "another example"],
        padding=True,
        return_tensors="pt",
    )
    dummy_input_ids = dummy_batch["input_ids"]
    dummy_attention_mask = dummy_batch["attention_mask"]
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        (dummy_input_ids, dummy_attention_mask),
        OUT_PATH,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    with torch.no_grad():
        torch_logits = wrapped(dummy_input_ids, dummy_attention_mask)

    session = ort.InferenceSession(OUT_PATH)
    (onnx_logits,) = session.run(
        OUTPUT_NAMES,
        {
            "input_ids": dummy_input_ids.numpy(),
            "attention_mask": dummy_attention_mask.numpy(),
        },
    )
    assert torch.allclose(torch.from_numpy(onnx_logits), torch_logits, atol=1e-4)


if __name__ == "__main__":
    export()

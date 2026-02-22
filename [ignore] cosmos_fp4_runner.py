"""
CosmosFP4Runner -- Hazard-perception inference with Cosmos-Reason1-7B NVFP4
via TensorRT-LLM on Blackwell GPUs.

Drop-in replacement for CosmosFP8Runner that uses FP4 Tensor Core GEMM
instead of FP8 bitsandbytes quantization.

Prerequisites:
    1. source activate_fp4.sh
    2. python3 quantize_cosmos_fp4.py   (creates the NVFP4 checkpoint)

Usage:
    from cosmos_reason1_fp4_inference import CosmosFP4Runner

    runner = CosmosFP4Runner(user_prompt)
    metrics = runner.run_inference_dataloader(
        dataloader, vid_output, vid_fps, vid_width, vid_height,
    )
"""

import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.metrics import Metrics
from utils.inference_video_writer import annotate_and_write_frames

from .cosmos_fp4_inference import (
    DEFAULT_MODEL,
    SOURCE_MODEL,
    verify_gpu,
    load_trtllm_model,
    load_video_frames,
    analyze_video,
    parse_result as _parse_result_str, # TODO Use independent func
)


class CosmosFP4Runner:
    """
    Cosmos-Reason1-7B NVFP4 runner for hazard-perception video classification.

    Uses TensorRT-LLM with FP4 Tensor Core GEMM on Blackwell GPUs.
    Provides the same dataloader interface as CosmosFP8Runner so the two
    backends can be swapped transparently.
    """

    def __init__(
        self,
        user_prompt: str,
        model_path: str = DEFAULT_MODEL,
        source_model: str = SOURCE_MODEL,
        target_fps: int = 4,
        max_tokens: int = 7,
        target_resolution: tuple[int, int] = (250, 250),
    ):
        self.user_prompt = user_prompt
        self.model_path = model_path
        self.source_model = source_model
        self.target_fps = target_fps
        self.max_tokens = max_tokens
        self.target_resolution = target_resolution

        self.llm = None
        self.processor = None

        verify_gpu()
        self.load_model()
        self.warmup_model()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load NVFP4 model via TRT-LLM and the processor from the source model."""
        print(f"Loading Cosmos-Reason1 NVFP4 model from {self.model_path} ...")
        start = time.time()

        self.llm = load_trtllm_model(self.model_path)

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            self.source_model, trust_remote_code=True,
        )

        print(f"Model + processor ready in {time.time() - start:.1f}s\n")

    def warmup_model(self) -> None:
        """Run a throwaway generation to compile TRT-LLM kernels."""
        if self.llm is None:
            raise ValueError("Model must be loaded before warmup.")

        print("Warming up TRT-LLM engine ...")
        from tensorrt_llm import SamplingParams

        self.llm.generate(
            ["Hello, describe this scene briefly."],
            SamplingParams(max_tokens=5),
            use_tqdm=False,
        )
        torch.cuda.synchronize()
        print("Warmup complete.\n")

    # ------------------------------------------------------------------
    # Dataloader-driven inference (matches CosmosFP8Runner interface)
    # ------------------------------------------------------------------

    def run_inference_dataloader(
        self,
        dataloader,
        vid_output: str | Path,
        vid_fps: float,
        vid_width: int,
        vid_height: int,
    ) -> Metrics:
        """
        Run FP4 inference over a hazard-perception dataloader.

        Args:
            dataloader: Iterable yielding ``(video_path, label, new_images)``
                tuples (same contract as ``HarzardPerceptionTestDataLoader``).
            vid_output: Path for the annotated output video.
            vid_fps: Frame rate of the output video.
            vid_width: Width of the output video frames.
            vid_height: Height of the output video frames.

        Returns:
            A populated ``Metrics`` instance with TP/TN/FP/FN and timing data.
        """
        preds: list[bool] = []
        actuals: list[bool] = []
        inference_times: list[float] = []

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(vid_output), fourcc, vid_fps, (vid_width, vid_height),
        )

        for i, (video_path, label, new_images) in enumerate(dataloader):
            infer_start = time.time()
            raw_output = self.process_and_run_single_video(video_path)
            infer_time = time.time() - infer_start

            anomaly = self.parse_result(raw_output)

            preds.append(anomaly)
            actuals.append(label)
            inference_times.append(infer_time)

            print(
                f"\t[{i + 1}] Actual: {label}, Predicted: {anomaly} "
                f"(FP4 Inference: {infer_time:.2f}s)"
            )

            annotate_and_write_frames(
                video_writer, new_images, anomaly, label, i, vid_width,
            )

        video_writer.release()
        print(f"Processed video saved at '{vid_output}'")

        self._print_metrics(preds, actuals)

        metrics = Metrics()
        metrics.update(preds, actuals, inference_times)
        return metrics

    # ------------------------------------------------------------------
    # Single-video inference
    # ------------------------------------------------------------------

    def process_and_run_single_video(self, video_path: str | Path) -> str:
        """Load and analyse a single video clip using FP4 inference."""
        video_path = Path(video_path)
        print(f"Processing video (FP4): {video_path}")

        prefetched = load_video_frames(
            video_path, self.target_fps, self.target_resolution,
        )
        return analyze_video(
            self.llm,
            self.processor,
            video_path,
            prefetched,
            self.max_tokens,
            self.user_prompt,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_result(raw_output: str) -> bool:
        """Map model output to a boolean label (True = Anomaly)."""
        return _parse_result_str(raw_output) == "Anomaly"
        # TODO TOO BRITTLE

    @staticmethod
    def _print_metrics(predictions, actuals) -> None:
        from sklearn.metrics import precision_score, recall_score, f1_score

        correct = sum(p == a for p, a in zip(predictions, actuals))
        accuracy = correct / len(actuals) if actuals else 0

        tp = sum(p and a for p, a in zip(predictions, actuals))
        fp = sum(p and not a for p, a in zip(predictions, actuals))
        tn = sum(not p and not a for p, a in zip(predictions, actuals))
        fn = sum(not p and a for p, a in zip(predictions, actuals))

        print("=" * 30)
        print(f"Accuracy:  {accuracy * 100:.2f}%")
        print(f"Precision: {precision_score(actuals, predictions, zero_division=0) * 100:.2f}%")
        print(f"Recall:    {recall_score(actuals, predictions, zero_division=0) * 100:.2f}%")
        print(f"F1-score:  {f1_score(actuals, predictions, zero_division=0) * 100:.2f}%")
        print(f"TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")

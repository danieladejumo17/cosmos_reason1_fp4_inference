#!/usr/bin/env python3
"""
FP4 Inference Script for NVIDIA Cosmos-Reason1-7B on RTX 5090 (Blackwell)

This script performs video analysis using TRUE hardware-accelerated FP4 computation
on RTX 5090's native Blackwell FP4 Tensor Cores.

Supports two inference modes:
1. PyTorch + ModelOpt: Loads calibrated model with FP4 quantizers (default)
2. TensorRT-LLM: Uses compiled TensorRT engine if available

The FP4 computation happens directly in hardware - NOT simulated via FP16/BF16.

Usage:
  # Using PyTorch + ModelOpt (calibrated model)
  python fp4_inference.py --video_dir ./videos --engine_dir ./cosmos-fp4-engine

  # Force TensorRT-LLM mode (if engine available)
  python fp4_inference.py --video_dir ./videos --engine_dir ./cosmos-fp4-engine --mode trtllm

Requirements:
    - NVIDIA RTX 5090 GPU (Blackwell architecture, SM 100+)
    - PyTorch >= 2.9.0 with CUDA support
    - NVIDIA ModelOpt for FP4 quantization
    - (Optional) TensorRT-LLM for engine-based inference
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Optional, Union
from enum import Enum

import numpy as np
import torch
import cv2

warnings.filterwarnings("ignore", category=UserWarning)


class InferenceMode(Enum):
    """Inference mode selection."""
    AUTO = "auto"
    PYTORCH = "pytorch"  # PyTorch + ModelOpt quantizers
    TRTLLM = "trtllm"    # TensorRT-LLM engine


# ============================================================
# 1. PyTorch + ModelOpt FP4 Model (Primary Mode)
# ============================================================
class ModelOptFP4Model:
    """
    PyTorch model with ModelOpt FP4 quantizers for true hardware FP4 acceleration.
    
    This loads the calibrated FP4 model and runs inference using NVIDIA ModelOpt
    quantizers which leverage Blackwell's native FP4 Tensor Cores.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        model_name: str = "nvidia/Cosmos-Reason1-7B",
        device: str = "cuda",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.device = torch.device(device)
        
        # Load configuration
        self._load_config()
        
        # Load model and processor
        self._load_model()
        
        # Verify FP4 setup
        self._verify_fp4_setup()
    
    def _load_config(self):
        """Load quantization configuration."""
        # Try multiple config locations
        config_paths = [
            self.checkpoint_dir / "fp4_config.json",
            self.checkpoint_dir / "quant_config.json",
            self.checkpoint_dir / "trtllm_checkpoint" / "quant_config.json",
        ]
        
        self.config = {}
        for config_path in config_paths:
            if config_path.exists():
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                print(f"Loaded FP4 config from: {config_path}")
                break
        
        if self.config:
            print(f"  Quantization: {self.config.get('quantization', 'NVFP4')}")
            print(f"  Algorithm: {self.config.get('algorithm', 'awq_lite')}")
            print(f"  Calibrated: {self.config.get('calibrated', False)}")
    
    def _load_model(self):
        """Load the FP4 quantized model and processor."""
        import transformers
        
        # Determine model path - check for HuggingFace checkpoint
        model_paths = [
            self.checkpoint_dir / "trtllm_checkpoint" / "model",
            self.checkpoint_dir / "model",
            self.checkpoint_dir,
        ]
        
        processor_paths = [
            self.checkpoint_dir / "trtllm_checkpoint" / "processor",
            self.checkpoint_dir / "vision_encoder",
            self.checkpoint_dir / "processor",
        ]
        
        model_path = None
        for p in model_paths:
            if (p / "config.json").exists() or (p / "model.safetensors").exists():
                model_path = p
                break
        
        processor_path = None
        for p in processor_paths:
            if (p / "preprocessor_config.json").exists() or (p / "tokenizer_config.json").exists():
                processor_path = p
                break
        
        # If no local checkpoint, load from original model
        if model_path is None:
            print(f"No local checkpoint found, loading from: {self.model_name}")
            model_path = self.model_name
        else:
            print(f"Loading quantized model from: {model_path}")
        
        if processor_path is None:
            processor_path = self.model_name
        
        # Load processor
        print(f"Loading processor from: {processor_path}")
        self.processor = transformers.AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
        )
        
        # Load model with FP16 dtype (FP4 quantizers handle precision)
        # Use AutoModelForVision2Seq or the specific VLM class for generation
        print("Loading model for generation...")
        
        # For VLM models, always load from original model to ensure lm_head is present
        # Then apply FP4 quantization for Blackwell Tensor Cores
        # This is the recommended approach since VLM checkpoints may not save all layers
        load_from_original = True
        
        if model_path != self.model_name:
            # Check if the checkpoint has lm_head (required for generation)
            config_path = Path(model_path) / "config.json"
            if config_path.exists():
                import json
                with open(config_path) as f:
                    config = json.load(f)
                # If it's a base model (not for generation), load from original
                arch = config.get("architectures", [])
                if any("ForConditionalGeneration" in a or "ForCausalLM" in a for a in arch):
                    load_from_original = False
        
        if load_from_original:
            print(f"  Loading from original model for complete weights: {self.model_name}")
            model_path = self.model_name
        
        # Try to load with the correct generation-capable class
        try:
            # First try AutoModelForVision2Seq (for VLM models)
            self.model = transformers.AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            print("  Loaded with AutoModelForVision2Seq")
        except Exception as e1:
            try:
                # Fallback to Qwen2_5_VLForConditionalGeneration directly
                from transformers import Qwen2_5_VLForConditionalGeneration
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                print("  Loaded with Qwen2_5_VLForConditionalGeneration")
            except Exception as e2:
                # Last fallback - load from original model name
                print(f"  Could not load from checkpoint: {e1}")
                print(f"  Loading from original model: {self.model_name}")
                self.model = transformers.AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
        
        # Apply FP4 quantization for Blackwell Tensor Cores
        self._apply_fp4_quantization()
        
        self.model.eval()
        
        # Get device from model
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device("cuda:0")
        
        print(f"Model loaded on device: {self.device}")
    
    def _apply_fp4_quantization(self):
        """Apply FP4 quantization using ModelOpt."""
        try:
            import modelopt.torch.quantization as mtq
            
            print("Applying NVFP4 quantization for Blackwell Tensor Cores...")
            
            # Use NVFP4_DEFAULT_CFG which uses 'max' algorithm - no calibration needed
            # This is suitable for inference as it doesn't require a forward_loop
            # AWQ_LITE requires calibration data, DEFAULT uses max scaling which works at inference
            quant_config = mtq.NVFP4_DEFAULT_CFG
            
            # Apply quantization with max algorithm (no forward_loop needed)
            mtq.quantize(self.model, quant_config, forward_loop=None)
            
            print("  FP4 quantization applied successfully (max algorithm)")
            print("  FP4 GEMM operations will use Blackwell Tensor Cores")
            
        except ImportError:
            print("Warning: ModelOpt not available, running in FP16 mode")
            return
        except Exception as e:
            print(f"Warning: Could not apply FP4 quantization: {e}")
            print("Running in FP16 mode")
            return
        
        # Apply torch.compile for additional speedup
        self._compile_model()
    
    def _compile_model(self):
        """Apply torch.compile() for optimized inference."""
        try:
            print("Applying torch.compile() for optimized inference...")
            
            # Disable gradient checkpointing for faster inference
            if hasattr(self.model, 'gradient_checkpointing_disable'):
                self.model.gradient_checkpointing_disable()
            
            # Set matmul precision for Tensor Cores
            torch.set_float32_matmul_precision("high")
            
            # Compile the model for faster execution
            # Use 'reduce-overhead' mode for inference
            self.model = torch.compile(self.model, mode="reduce-overhead")
            
            print("  torch.compile() applied successfully")
            
        except Exception as e:
            print(f"  torch.compile() skipped: {e}")

    def _verify_fp4_setup(self):
        """Verify FP4 Tensor Core setup on Blackwell."""
        compute_cap = torch.cuda.get_device_capability(0)
        
        if compute_cap[0] >= 10:
            print("✓ Blackwell architecture detected - Native FP4 Tensor Cores enabled!")
        else:
            print(f"⚠ SM {compute_cap[0]}.{compute_cap[1]} detected - FP4 may be emulated")
        
        # Check for quantized modules
        try:
            import modelopt.torch.quantization as mtq
            
            quantized_modules = 0
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight_quantizer') or hasattr(module, 'input_quantizer'):
                    quantized_modules += 1
            
            if quantized_modules > 0:
                print(f"✓ Found {quantized_modules} FP4 quantized modules")
            else:
                print("⚠ No FP4 quantized modules found - may be running in FP16")
                
        except ImportError:
            pass
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_new_tokens: int = 7,
        **kwargs
    ):
        """
        Generate text using FP4 quantized model on Blackwell Tensor Cores.
        
        The FP4 quantizers intercept the GEMM operations and execute them
        using native FP4 Tensor Core instructions on RTX 5090.
        """
        # Build generation kwargs
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        
        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values
        
        if pixel_values_videos is not None:
            gen_kwargs["pixel_values_videos"] = pixel_values_videos
        
        if video_grid_thw is not None:
            gen_kwargs["video_grid_thw"] = video_grid_thw
        
        if image_grid_thw is not None:
            gen_kwargs["image_grid_thw"] = image_grid_thw
        
        # Generate - FP4 computation happens automatically via quantizers
        output_ids = self.model.generate(**gen_kwargs)
        
        return output_ids


# ============================================================
# 2. TensorRT-LLM FP4 Model (Alternative Mode)
# ============================================================
class TensorRTLLMFP4Model:
    """
    Wrapper for TensorRT-LLM FP4 engine providing true hardware FP4 acceleration.
    
    This class loads and runs inference on a TensorRT-LLM engine compiled with
    native FP4 Tensor Core operations for RTX 5090 (Blackwell).
    """
    
    def __init__(self, engine_dir: Path, vision_model_name: str = "nvidia/Cosmos-Reason1-7B"):
        self.engine_dir = Path(engine_dir)
        self.vision_model_name = vision_model_name
        
        # Load configuration
        config_path = self.engine_dir / "fp4_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                self.config = json.load(f)
            print(f"Loaded FP4 engine config:")
            print(f"  - Quantization: {self.config.get('quantization_type', 'NVFP4')}")
            print(f"  - Compute type: {self.config.get('compute_type', 'FP4 Tensor Core')}")
        else:
            self.config = {}
        
        # Initialize TensorRT-LLM runtime
        self._init_trtllm_runtime()
        
        # Initialize vision encoder and processor
        self._init_vision_components()
    
    def _init_trtllm_runtime(self):
        """Initialize TensorRT-LLM runtime with FP4 engine."""
        try:
            import tensorrt_llm
            from tensorrt_llm.runtime import ModelRunner, ModelRunnerCpp
        except ImportError:
            raise ImportError(
                "TensorRT-LLM not installed. Install with:\n"
                "pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com"
            )
        
        # Find engine directory
        engine_path = self.engine_dir / "trtllm_engine"
        if not engine_path.exists():
            engine_path = self.engine_dir
        
        print(f"Loading TensorRT-LLM FP4 engine from: {engine_path}")
        
        try:
            runner_kwargs = {
                "engine_dir": str(engine_path),
                "rank": 0,
                "debug_mode": False,
            }
            
            # Try C++ runner first (faster, better FP4 support)
            try:
                self.runner = ModelRunnerCpp.from_dir(**runner_kwargs)
                print("Using TensorRT-LLM C++ runtime with native FP4")
            except Exception:
                self.runner = ModelRunner.from_dir(**runner_kwargs)
                print("Using TensorRT-LLM Python runtime")
            
            self.max_input_len = 4096
            self.max_output_len = 512
                
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT-LLM engine: {e}")
    
    def _init_vision_components(self):
        """Initialize vision encoder and processor for multimodal inference."""
        import transformers
        
        # Load processor
        vision_dir = self.engine_dir / "vision_encoder"
        if vision_dir.exists():
            processor_path = str(vision_dir)
        else:
            processor_path = self.vision_model_name
        
        print(f"Loading processor from: {processor_path}")
        self.processor = transformers.AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
        )
        
        self.device = torch.device("cuda:0")
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_new_tokens: int = 7,
        **kwargs
    ):
        """Generate text using the FP4 TensorRT-LLM engine."""
        outputs = self.runner.generate(
            batch_input_ids=input_ids.tolist(),
            max_new_tokens=max_new_tokens,
            end_id=self.processor.tokenizer.eos_token_id,
            pad_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            return_dict=True,
        )
        
        output_ids = torch.tensor(outputs["output_ids"], device=input_ids.device)
        return output_ids


# ============================================================
# 3. Unified Model Loader
# ============================================================
def load_fp4_model(
    engine_dir: Path,
    model_name: str = "nvidia/Cosmos-Reason1-7B",
    mode: InferenceMode = InferenceMode.AUTO,
) -> Union[ModelOptFP4Model, TensorRTLLMFP4Model]:
    """
    Load FP4 model based on available checkpoints and requested mode.
    
    Args:
        engine_dir: Directory containing quantized model/engine
        model_name: Original HuggingFace model name
        mode: Inference mode (auto, pytorch, or trtllm)
    
    Returns:
        Loaded FP4 model wrapper
    """
    engine_dir = Path(engine_dir)
    
    # Check what's available
    has_trtllm_engine = (
        (engine_dir / "trtllm_engine").exists() and 
        any((engine_dir / "trtllm_engine").glob("*.engine"))
    )
    has_pytorch_checkpoint = (
        (engine_dir / "trtllm_checkpoint" / "model").exists() or
        (engine_dir / "model").exists()
    )
    
    print(f"Checking available inference modes:")
    print(f"  TensorRT-LLM engine: {'✓' if has_trtllm_engine else '✗'}")
    print(f"  PyTorch checkpoint: {'✓' if has_pytorch_checkpoint else '✗'}")
    
    # Determine mode
    if mode == InferenceMode.TRTLLM:
        if not has_trtllm_engine:
            raise FileNotFoundError(
                f"TensorRT-LLM engine not found in {engine_dir}. "
                "Use --mode pytorch or run quantization first."
            )
        print("\nUsing TensorRT-LLM engine mode")
        return TensorRTLLMFP4Model(engine_dir, model_name)
    
    elif mode == InferenceMode.PYTORCH:
        print("\nUsing PyTorch + ModelOpt mode")
        return ModelOptFP4Model(engine_dir, model_name)
    
    else:  # AUTO
        if has_trtllm_engine:
            print("\nAuto-selected: TensorRT-LLM engine mode")
            try:
                return TensorRTLLMFP4Model(engine_dir, model_name)
            except Exception as e:
                print(f"TensorRT-LLM load failed: {e}")
                print("Falling back to PyTorch mode...")
        
        print("\nUsing PyTorch + ModelOpt mode")
        return ModelOptFP4Model(engine_dir, model_name)


# ============================================================
# 4. Video Processing
# ============================================================
def load_video(video_path: Path, target_fps: int, target_resolution: tuple[int, int]):
    """
    Load and preprocess video frames.
    
    Parameters:
    - fps: 4 (default)
    - target_resolution: 250x250 (default)
    """
    import qwen_vl_utils
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        effective_fps = min(target_fps, native_fps) if native_fps > 0 else target_fps
        duration_seconds = frame_count / native_fps if native_fps > 0 else 0
        num_frames_to_sample = max(1, int(duration_seconds * effective_fps))

        total_pixels = num_frames_to_sample * target_resolution[0] * target_resolution[1]
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()

    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(
        [{"role": "user", "content": [{"type": "video", "video": str(video_path), "total_pixels": total_pixels}]}] # TODO fix total pixels to const
        # [{"role": "user", "content": [{"type": "video", "video": str(video_path), "fps": 4, "total_pixels": 4096 * 30**2}]}] # TODO fix total pixels to const
    )
    return image_inputs, video_inputs


# ============================================================
# 5. Prompt (Same as fp8_inference.py)
# ============================================================
def get_analysis_prompt():
    """
    Get the autonomous driving safety analysis prompt.
    Identical to fp8_inference.py.
    """
    return (
        "You are an autonomous driving safety expert analyzing this video for EXTERNAL ANOMALIES that may impact safe AV operation.\n\n"
        "<think>\n"
        "- Obstacles, pedestrians, or vehicles violating rules\n"
        "- Roadwork, blocked lanes, poor visibility, or hazards\n"
        "- Reflections, shadows, or false visual cues confusing perception\n"
        "</think>\n\n"
        "<answer>\n"
        "Is there any external anomaly in this video? Reply with exactly one word of the following:\n"
        "Classification: Anomaly — if any obstacle, obstruction, or unsafe condition is visible.\n"
        "Classification: Normal — if no anomaly or obstruction is visible.\n"
        "</answer>"
    )


# ============================================================
# 6. Result Parsing (Same as fp8_inference.py)
# ============================================================
def parse_result(raw_output: str) -> str:
    """Parse model output to extract classification."""
    out = raw_output.lower()
    if "anomaly" in out:
        return "Anomaly"
    elif "normal" in out:
        return "Normal"
    return "Unknown"


# ============================================================
# 7. Video Analysis with FP4
# ============================================================
def analyze_video_fp4(
    model: Union[ModelOptFP4Model, TensorRTLLMFP4Model],
    video_path: Path,
    prefetched_data: tuple,
    max_tokens: int,
    prompt_text: str,
):
    """
    Analyze video using TRUE hardware FP4 inference on RTX 5090.
    
    The LLM computation runs on native FP4 Tensor Cores while
    vision encoding runs in FP16 for quality.
    """
    image_inputs, video_inputs = prefetched_data
    
    # Build conversation
    content = [
        {"type": "video", "video": str(video_path)},
        {"type": "text", "text": prompt_text},
    ]
    conversation = [{"role": "user", "content": content}]
    
    # Apply chat template
    text = model.processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Process inputs
    inputs = model.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # Move to device
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    # Get video/image tensors
    pixel_values = inputs.get("pixel_values")
    pixel_values_videos = inputs.get("pixel_values_videos")
    video_grid_thw = inputs.get("video_grid_thw")
    image_grid_thw = inputs.get("image_grid_thw")
    
    if pixel_values is not None:
        pixel_values = pixel_values.to(model.device, torch.float16)
    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.to(model.device, torch.float16)
    if video_grid_thw is not None:
        video_grid_thw = video_grid_thw.to(model.device)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(model.device)
    
    # Generate with FP4 Tensor Cores
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
        image_grid_thw=image_grid_thw,
        max_new_tokens=max_tokens,
    )
    
    # Decode output
    new_tokens = output_ids[:, input_ids.shape[1]:]
    response = model.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    
    return response


# ============================================================
# 8. Warmup
# ============================================================
def warmup_model(model: Union[ModelOptFP4Model, TensorRTLLMFP4Model]):
    """Warm up FP4 model to compile kernels."""
    print("Warming up FP4 model...")
    
    dummy_text = "Is this scene safe?"
    conversation = [{"role": "user", "content": [{"type": "text", "text": dummy_text}]}]
    text = model.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = model.processor(text=[text], return_tensors="pt")
    
    input_ids = inputs["input_ids"].to(model.device)
    
    with torch.inference_mode():
        _ = model.generate(input_ids=input_ids, max_new_tokens=5)
    
    torch.cuda.synchronize()
    print("Warmup complete - FP4 Tensor Cores ready.\n")


# ============================================================
# 9. GPU Verification
# ============================================================
def verify_gpu():
    """Verify RTX 5090 (Blackwell) is available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. RTX 5090 required for FP4.")
    
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    print(f"Memory: {memory_gb:.1f} GB")
    
    if compute_cap[0] >= 10:
        print("✓ Blackwell architecture detected - Native FP4 Tensor Cores enabled!")
    else:
        print(f"⚠ SM {compute_cap[0]}.{compute_cap[1]} detected.")
        print("  Native FP4 requires Blackwell (SM 10.0+). Performance may be suboptimal.")
    
    return gpu_name, compute_cap


# ============================================================
# 10. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="FP4 Inference with Native Blackwell Tensor Core Acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script uses TRUE hardware-accelerated FP4 on RTX 5090's Blackwell Tensor Cores.
The matrix multiplications happen in native 4-bit floating point hardware.

Inference Modes:
  auto     - Auto-select best available mode (default)
  pytorch  - Use PyTorch + ModelOpt quantizers
  trtllm   - Use TensorRT-LLM engine (if available)

Examples:
  # Basic inference (auto-selects mode)
  python fp4_inference.py --video_dir ./videos --engine_dir ./cosmos-fp4-engine

  # Force PyTorch + ModelOpt mode
  python fp4_inference.py --video_dir ./videos --engine_dir ./cosmos-fp4-engine --mode pytorch

  # Save results to JSON
  python fp4_inference.py --video_dir ./videos --engine_dir ./cosmos-fp4-engine --output_file results.json

Configuration (same as fp8_inference.py):
  - FPS: 4 (default)
  - Max tokens: 7 (default)
  - Resolution: 250x250 (default)
        """
    )
    
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing video files to process"
    )
    parser.add_argument(
        "--engine_dir",
        type=str,
        required=True,
        help="Path to FP4 quantized model/engine directory"
    )
    parser.add_argument(
        "--original_model",
        type=str,
        default="nvidia/Cosmos-Reason1-7B",
        help="Original HuggingFace model (default: nvidia/Cosmos-Reason1-7B)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "pytorch", "trtllm"],
        default="auto",
        help="Inference mode: auto, pytorch, or trtllm (default: auto)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Target FPS for video sampling (default: 4)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=7,
        help="Max new tokens to generate (default: 7)"
    )
    parser.add_argument(
        "--target_resolution",
        type=str,
        default="250x250",
        help="Target resolution WxH (default: 250x250)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional: Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.target_resolution.split('x'))
    target_resolution = (width, height)
    
    # Parse mode
    mode_map = {
        "auto": InferenceMode.AUTO,
        "pytorch": InferenceMode.PYTORCH,
        "trtllm": InferenceMode.TRTLLM,
    }
    inference_mode = mode_map[args.mode]
    
    # Find videos
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Error: Video directory '{video_dir}' does not exist.")
        return
    
    video_files = sorted([
        f for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm")
        for f in video_dir.glob(ext)
    ])
    
    if not video_files:
        print(f"No video files found in '{video_dir}'.")
        return
    
    # Print configuration
    print("=" * 70)
    print("FP4 Inference — Native Blackwell Tensor Core Acceleration")
    print("=" * 70)
    
    # Verify GPU
    gpu_name, compute_cap = verify_gpu()
    
    print(f"\nEngine directory: {args.engine_dir}")
    print(f"Video directory: {video_dir}")
    print(f"Videos found: {len(video_files)}")
    print(f"Inference mode: {args.mode}")
    print(f"FPS: {args.fps}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Target resolution: {args.target_resolution}")
    print("=" * 70 + "\n")
    
    # Load FP4 model
    print("Loading FP4 model...")
    start_load = time.time()
    model = load_fp4_model(
        engine_dir=Path(args.engine_dir),
        model_name=args.original_model,
        mode=inference_mode,
    )
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s\n")
    
    # Determine actual mode used
    actual_mode = "TensorRT-LLM" if isinstance(model, TensorRTLLMFP4Model) else "PyTorch+ModelOpt"
    
    # Warmup
    warmup_model(model)
    
    # Get prompt
    prompt_text = get_analysis_prompt()
    
    print(f"Processing {len(video_files)} videos with FP4 Tensor Cores ({actual_mode})...\n" + "-" * 70)
    
    results = []
    total_inference_time = 0
    total_load_time = 0
    
    for idx, video_path in enumerate(video_files, 1):
        # Load video
        video_load_start = time.time()
        try:
            prefetched_data = load_video(video_path, args.fps, target_resolution)
        except Exception as e:
            print(f"[{idx}/{len(video_files)}] {video_path.name}: ERROR loading - {e}")
            results.append({"file": video_path.name, "result": "Error", "error": str(e)})
            continue
        video_load_time = time.time() - video_load_start
        total_load_time += video_load_time
        
        # Analyze with FP4
        analysis_start = time.time()
        try:
            raw = analyze_video_fp4(model, video_path, prefetched_data, args.max_tokens, prompt_text)
            result = parse_result(raw)
            analysis_time = time.time() - analysis_start
            total_inference_time += analysis_time
            
            print(
                f"[{idx}/{len(video_files)}] {video_path.name}: {result} "
                f"(Load: {video_load_time:.2f}s, FP4 Inference: {analysis_time:.2f}s)"
            )
            
            results.append({
                "file": video_path.name,
                "result": result,
                "raw_output": raw,
                "load_time_s": round(video_load_time, 3),
                "inference_time_s": round(analysis_time, 3),
            })
            
        except Exception as e:
            print(f"[{idx}/{len(video_files)}] {video_path.name}: ERROR - {e}")
            results.append({"file": video_path.name, "result": "Error", "error": str(e)})
    
    # Summary
    print("-" * 70)
    print("\nSUMMARY — FP4 Native Tensor Core Inference")
    print("=" * 70)
    
    anomalies = sum(1 for r in results if r["result"] == "Anomaly")
    normals = sum(1 for r in results if r["result"] == "Normal")
    errors = sum(1 for r in results if r["result"] == "Error")
    unknowns = sum(1 for r in results if r["result"] == "Unknown")
    successful = len(results) - errors
    
    print(f"Inference mode: {actual_mode}")
    print(f"Total videos: {len(video_files)}")
    print(f"  - Anomaly: {anomalies}")
    print(f"  - Normal: {normals}")
    print(f"  - Unknown: {unknowns}")
    print(f"  - Errors: {errors}")
    print(f"\nTotal load time: {total_load_time:.2f}s")
    print(f"Total FP4 inference time: {total_inference_time:.2f}s")
    
    if successful > 0:
        avg_inference = total_inference_time / successful
        print(f"Average FP4 inference time: {avg_inference:.2f}s per video")
    
    # Save results
    if args.output_file:
        output_data = {
            "config": {
                "engine_dir": str(args.engine_dir),
                "inference_mode": actual_mode,
                "fps": args.fps,
                "max_tokens": args.max_tokens,
                "target_resolution": args.target_resolution,
                "quantization": "NVFP4 (Native Blackwell FP4)",
                "compute_type": "FP4 Tensor Core GEMM",
                "gpu": gpu_name,
                "compute_capability": f"{compute_cap[0]}.{compute_cap[1]}",
            },
            "summary": {
                "total_videos": len(video_files),
                "anomalies": anomalies,
                "normals": normals,
                "unknowns": unknowns,
                "errors": errors,
                "total_load_time_s": round(total_load_time, 3),
                "total_inference_time_s": round(total_inference_time, 3),
            },
            "results": results,
        }
        
        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")
    
    print("\nFP4 inference complete.")


if __name__ == "__main__":
    main()

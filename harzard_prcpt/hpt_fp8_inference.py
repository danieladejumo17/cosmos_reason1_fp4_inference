import os
os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'

import cv2
from pathlib import Path

from hpt_dataLoader import HarzardPerceptionTestDataLoader
from utils.metrics import Metrics

# Run Inference on Hazard Perception Test Data Loader using Cosmos FP8 Runner
# Change working directory into the harzard_prcpt folder first
if __name__ == "__main__":
    import sys as _sys

    _PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
    if _PROJECT_ROOT not in _sys.path:
        _sys.path.insert(0, _PROJECT_ROOT)

    from cosmos_reason1_inference import CosmosFP8Runner
    import qwen_vl_utils

    dataloader = HarzardPerceptionTestDataLoader("./videos", "labels.csv", min_hazard_frames=10, out_video_folder="./fp8_output/output_videos", separate_videos=True)
    user_prompt = (
                "You are an autonomous driving safety expert analyzing this video for EXTERNAL ANOMALIES that may impact safe AV operation.\n\n"
                "<think>\n"
                "Focus on:\n"
                "- Obstacles, pedestrians, or vehicles violating rules\n"
                "- Roadwork, blocked lanes, poor visibility, or hazards\n"
                "- Reflections, shadows, or false visual cues confusing perception\n"
                "</think>\n\n"
                "<answer>\n"
                "Is there any external anomaly in this video? Reply with exactly one of the following:\n"
                "Classification: Anomaly — if any obstacle, obstruction, or unsafe condition is visible.\n"
                "Classification: Normal — if no anomaly or obstruction is visible.\n"
                "</answer>"
            )

    runner = CosmosFP8Runner(user_prompt)
    output_root = Path("./fp8_output")
    output_root.mkdir(exist_ok=True)

    # Extract video fps, width, height from first video in dataloader
    first_video_path, _, _ = next(iter(dataloader))
    cap = cv2.VideoCapture(first_video_path)
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"\n\nVideo FPS: {vid_fps}, Width: {vid_width}, Height: {vid_height}")

    # reset dataloader
    dataloader = HarzardPerceptionTestDataLoader("./videos", "labels.csv", min_hazard_frames=10, out_video_folder="./fp8_output/output_videos", separate_videos=True)

    # run inference
    metrics : Metrics = runner.run_inference_dataloader(
        dataloader,
        vid_output=output_root / "inference_results.mp4",
        vid_fps=vid_fps,
        vid_width=vid_width,
        vid_height=vid_height,
    )

    # print metrics
    # print(metrics.compute())
    
    # # plot Confusion Matrix
    # metrics.plot_confusion_matrix()


    # def process_and_run_single_video(self, video_path):
    # runner.conversation[0]["content"][0]["video"] = str("/home/daniel/Dev/ms_proj/stu_dataset/output_video.mp4")
    # print(runner.conversation)

    # text = runner.processor.apply_chat_template(
    #     runner.conversation, tokenize=False, add_generation_prompt=True
    # )

    # image_inputs, video_inputs = qwen_vl_utils.process_vision_info(runner.conversation)

    # inputs = runner.processor(
    #     text=[text],
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )
    # inputs = inputs.to(runner.model.device)

    # generated_ids = runner.model.generate(**inputs, max_new_tokens=8)

    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :]
    #     for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    # ]
    # output_text = runner.processor.batch_decode(
    #     generated_ids_trimmed,
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=False,
    # )

    # print(output_text[0])
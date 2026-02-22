import cv2
import numpy as np


def annotate_and_write_frames(
    writer,
    images,
    predicted: bool,
    actual: bool,
    sample_idx: int,
    vid_width: int,
) -> None:
    """Annotate frames with prediction/actual labels and write to a VideoWriter."""
    for img in images:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        rect_color = (0, 255, 0) if predicted == actual else (0, 0, 255)
        cv2.rectangle(
            img_cv,
            (vid_width // 2 - 400, 30),
            (vid_width // 2 + 400, 140),
            rect_color,
            -1,
        )
        cv2.putText(
            img_cv,
            f"Predicted: {predicted} | Actual: {actual}",
            (vid_width // 2 - 360, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            img_cv,
            f"Sample Num: {sample_idx + 1}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        writer.write(img_cv)

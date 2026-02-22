from hpt_dataLoader import HarzardPerceptionTestDataLoader
from pathlib import Path



def main():
    # HarzardPerceptionTestDataLoader(
    #         self,
    #         video_folder: str | Path,
    #         labels_csv_file: str | Path,
    #         min_hazard_frames: int = 10,
    #         out_video_folder: str | Path = None,
    #         separate_videos: bool = False,
    #     )
    SCRIPT_DIR = Path(__file__).resolve().parent

    # make the output videos dir
    out_video_folder = SCRIPT_DIR/"data/hpt_5s_videos"
    out_video_folder.mkdir(exist_ok=True, parents=True)

    # Create the Dataloader object
    dl = HarzardPerceptionTestDataLoader(
            SCRIPT_DIR/"data/videos_subsample", SCRIPT_DIR/"data/labels.csv",
            min_hazard_frames=10,
            out_video_folder=SCRIPT_DIR/"data/hpt_5s_videos",
            separate_videos=True,
        )

    # Iterate through the generator
    for _ in dl:
        pass


if __name__ == "__main__":
    main()
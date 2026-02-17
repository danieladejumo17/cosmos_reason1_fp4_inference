# Download the dataset
cd stu_dataset/
mkdir -p data/
wget https://huggingface.co/datasets/danieladejumo/stu_dataset/resolve/main/sampledata/scene_125.tar.gz
tar -xzf scene_125.tar.gz -C data/
# rm scene_125.tar.gz

# Create a virtual environment
python3 -m venv .stu_venv
source .stu_venv/bin/activate

# Install the dependencies
pip install -r requirements.txt

# Run the dataset
# python3 stu_video_dataset.py
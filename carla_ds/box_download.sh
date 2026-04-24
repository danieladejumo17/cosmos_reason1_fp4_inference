#!/bin/bash

# Array of tokens extracted from your Box shared links
tokens=(
    "bye8mennt7osqm9lfbymk1n4rn4nd8e9"
    "j99gpe8t7hplg0buqg2il9ruywdbmlzt"
    "ockbnyuq2akdxunnt5l4f8rxnb4tfz7c"
    "lnasivs793wp451faxsq1oupzxspc0z9"
    "aj1tika40u8obilywpvxqw30uemdw642"
    "mcbezvcyxbhqj9zjeqwgi085w1onon5n"
    "3hq951a5mnxcc3ecuergn3524bhiezfw"
    "j3vxwcx404tr1ikylw1cn8w9ohyc7kpq"
    "3i1zlyqy26mldea1rm0lojvmo8eda0qa"
    "udchkwwjmwuqq6qxnsdz2e5rj8mgdazd"
)

echo "Starting download of 10 datasets..."

# Loop through the tokens
for i in "${!tokens[@]}"; do
    # Calculate file number (array index + 1)
    file_num=$((i + 1))
    filename="dataset_output_${file_num}.zip"
    token="${tokens[$i]}"
    
    # Construct the direct download URL
    # Format: https://nyu.box.com/shared/static/[TOKEN]
    url="https://nyu.box.com/shared/static/${token}"
    
    echo "------------------------------------------"
    echo "Downloading ${filename}..."
    
    # wget -O specifies the output filename
    # --continue allows resuming if the connection drops
    wget -O "$filename" "$url"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download ${filename}"
    fi
done

echo "------------------------------------------"
echo "All downloads complete. Starting extraction..."

# Unzip each file and remove the zip afterward
for zip_file in dataset_output_*.zip; do
    if [ -f "$zip_file" ]; then
        echo "Extracting $zip_file..."
        # -o overwrites existing files without prompting
        unzip -q -o "$zip_file"
        
        echo "Removing $zip_file..."
        rm "$zip_file"
    fi
done

echo "------------------------------------------"
echo "Done! All files unzipped and cleaned up."
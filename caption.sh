#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=16GB

module load python3.7-anaconda
module load tensorflow/1.15.5

# Sanity check command line options
usage() {
  echo "Usage: $0 (extract_data|delete_data|train)"
}

if [ $# < 1 ]; then
  usage
  exit 1
fi

# Parse argument.  $1 is the first argument
case $1 in
  "extract_data")
    if [ -d "data" ] 
    then
        echo "Directory data/ already exists"
    else
        unzip "data/test2014.zip" -d data
        unzip "data/train2014.zip" -d data
        unzip "data/val2014.zip" -d data
        unzip "data/annotations_trainval2014.zip" -d data
        unzip "data/image_info_test2014.zip" -d data
    fi
    ;;
    
  "download_data")
    wget -P data/ http://images.cocodataset.org/zips/train2014.zip
    wget -P data/ http://images.cocodataset.org/zips/val2014.zip
    wget -P data/ http://images.cocodataset.org/zips/test2014.zip
    wget -P data/ http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    wget -P data/ http://images.cocodataset.org/annotations/image_info_test2014.zip
    ;;

  "eval")
    python 595_final_project.py evaluate $2 $3 ${4:-foo}
    ;;

  "train")
    python 595_final_project.py train $2 $3 ${4:-foo} ${5:-foo}
    ;;
    
  "bleu")
    python 595_final_project.py bleu $2 $3 $4 $5
    ;;
    
  "train_mmbert")
    cd MMBERT/pretrain
    python roco_train.py --mlm_prob 0.15
    ;;
  
  "test_mmbert")
    cd MMBERT/pretrain
    python roco_test.py
    ;;
  *)
    usage
    exit 1
    ;;
esac

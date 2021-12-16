Automatic Image Captioning
==========================

EECS 595: Natural Language Processing, Fall 2021

University of Michigan - Ann Arbor

Group 29: Q Cong (congq@umich.edu), Rohan Naik (rgnaik@umich.edu)

# Instructions

## Setting up the dependencies

You can create a virtual environment for Python:

```
virtualenv venv
source venv/bin/activate
```

Installing the dependencies:

```
pip install -r requirements.txt
```

## Downloading and extracing the data

```
./caption.sh download_data
./caption.sh extract_data
```

"Invalid input. Run the code using one of the following formats:\n"
"1)\t train [cnn_model = {vgg, resnet, squeezenet}] [rnn_model = {gru, lstm}] [optional: path to training features] [optional: path to validation features]\n"
"2)\t evaluate [cnn_model = {vgg, resnet, squeezenet}] [path to trained_dict] [optional: path to testing features]\n\n"
"2)\t bleu [path to trained_dict] [path to features] [path to captions] [sample size]\n\n"
"For example, to train on ResNet and LSTM, use\n"
"train resnet lstm"

## Working with model

From now, we will refer to:
* `cnn` as one of the elements in the set `{vgg, resnet}`
* `rnn` as one of the elements in the set `{lstm, gru}`

**NOTE:**
* You can download all the [extracted features](https://drive.google.com/drive/folders/1kNOISCfXQJKkcgia9JteZJCAwC1VG41g?usp=sharing) (for training, testing, and validating) for each model as pickle files and place them under `features/`, so you can just use these to save time.
* You can also download the [trained model dicts](https://drive.google.com/drive/folders/1CcGLSNILf3-Q9HT9NZENDVTDCVO35R35?usp=sharing) and place them under `trained/`

### Training

The code format is

```
./caption.sh train cnn rnn (optional: path_to_training_features) (optional: path_to_validation_features)
```

For example, to train the ResNet50/LSTM model, run:

```
./caption.sh train resnet lstm
```

After training is complete, the model will be saved under `trained/trained_model_dict.pickle`. Use this pickle file for evaluating/testing. The training and validation features are also saved under `features/` for future use. For example, if you need to train another ResNet50 model, you can reuse the saved features so that we don't have to go through feature extraction another time.

Training will also output a training/validation loss plot in the main directory.


### Generating predictions

The code format is

```
./caption.sh evaluate cnn path_to_trained_dict (optional: path_to_testing_features)
```

For example, after training the ResNet50/LSTM model as above (and obtaining `trained/trained_model_dict.pickle`), you can generate some predictions by running:

```
./caption.sh evaluate resnet trained/trained_model_dict.pickle
```

Same as above, the testing features are also saved under `features/` for future use. For example, if you need to train another ResNet50 model, you can reuse the saved features so that we don't have to go through feature extraction another time.

The predictions will be outputted as `predictions.png` in the main directory.

### Evaluating with BLEU and METEOR Scores

The code format is

```
./caption.sh bleu path_to_trained_dict path_to_features path_to_captions sample_size
```

For example, after training the ResNet50/LSTM model as above (and obtaining `trained/trained_model_dict.pickle`), you can evaluate it by running:

```
./caption.sh bleu trained/trained_model_dict.pickle features/resnet/val_features_vgg.pickle data/annotations/captions_val2014.json 40000
```

The scores will be printed in stdout.


### Train and Test MMBERT

The codes we use are from [MMBERT: Multimodal BERT Pretraining for Improved Medical VQA
]{https://github.com/VirajBagal/MMBERT} with modifications to work on our data.

```
./caption.sh train_mmbert
./caption.sh test_mmbert
```

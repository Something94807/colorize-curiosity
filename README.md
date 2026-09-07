# Colorize Curiosity U-Net

A machine learning pipeline that colorizes raw, grayscale `.IMG` files from the NASA Curiosity Rover's Navcam and Hazcam, using a custom PyTorch U-Net. 

This repository includes the standalone web application, custom JPL API scraping tools, and training scripts used to build the model from scratch.

## Running the Web App

The main application is a Gradio web interface that natively parses raw `.IMG` files into LAB color space and applies the trained U-Net weights.

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   python app.py
   ``` 

### 1. Building the Dataset
To train the model, we need paired grayscale and color images of the Martian surface. `build_synthetic_pairs.py` is a concurrent scraper that queries the NASA JPL Planetary Data System (PDS) for Mastcam and MAHLI instruments. It filters out raw EDRs and blurry images, downloads valid color products, and synthetically generates corresponding grayscale inputs.

To generate a dataset of 2,000 image pairs:
```bash
cd training_pipeline
python build_synthetic_pairs.py --count 2000 --out-dir mars_dataset --workers 12
```
*Note: The script automatically generates a few side-by-side preview images in a `preview/` folder to verify data quality.*

### 2. Training the U-Net
`train.py` trains the U-Net model using L1 loss on the *a* and *b* color channels in CIELAB color space. It includes data augmentation such as random crops, flips, color jitter, and supports Automatic Mixed Precision (AMP) for faster GPU training.

To train from scratch:
```bash
python train.py --data-dir mars_dataset --epochs 30 --batch-size 8
```

To resume training from the latest checkpoint or fine-tune the model on a specialized dataset with a lower learning rate:
```bash
python train.py --data-dir mars_dataset --resume --lr 1e-5
```
During training, the script outputs validation grids to a `samples/` directory every 5 epochs so you can monitor colorization accuracy. the accuracy number is an estimate and should not be used as definitive proof of success. 

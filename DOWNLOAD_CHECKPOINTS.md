# How to Download Model Checkpoints from GCP

This guide explains how to download all 20 trained model checkpoints and the vocabulary file from your GCP instance.

## File Locations on GCP Instance

**GCP Instance Details:**
- Project: `pivotal-purpose-477111-r3`
- Zone: `asia-southeast1-a`
- Instance: `instance-20251103-132126`

**Checkpoint Files Location:**
```
/mnt/data/models/clean_cnn_transformer/
├── epoch_1.pth
├── epoch_2.pth
├── epoch_3.pth
├── epoch_4.pth
├── epoch_5.pth
├── epoch_6.pth
├── epoch_7.pth
├── epoch_8.pth
├── epoch_9.pth
├── epoch_10.pth
├── epoch_11.pth
├── epoch_12.pth
├── epoch_13.pth
├── epoch_14.pth
├── epoch_15.pth
├── epoch_16.pth
├── epoch_17.pth
├── epoch_18.pth
├── epoch_19.pth
└── epoch_20.pth
```

**Vocabulary File Location:**
```
/mnt/data/image-captioning-neural-networks/data/vocabulary.pkl
```

## Download Methods

### Method 1: Using GCP Console (Easiest)

1. Go to [GCP SSH Console](https://ssh.cloud.google.com/v2/ssh/projects/pivotal-purpose-477111-r3/zones/asia-southeast1-a/instances/instance-20251103-132126)
2. Click "DOWNLOAD FILE" button
3. Enter the file path (e.g., `/mnt/data/models/clean_cnn_transformer/epoch_20.pth`)
4. Click "Download"
5. Repeat for all 20 checkpoints and the vocabulary file

### Method 2: Using gcloud CLI (Recommended for Bulk Download)

**Step 1: Install gcloud CLI**
```bash
# Follow instructions at: https://cloud.google.com/sdk/docs/install
```

**Step 2: Authenticate**
```bash
gcloud auth login
gcloud config set project pivotal-purpose-477111-r3
```

**Step 3: Download All Checkpoints**
```bash
# Create local directory
mkdir -p ~/imagecaptioning-checkpoints

# Download all checkpoint files
for i in {1..20}; do
  gcloud compute scp \
    instance-20251103-132126:/mnt/data/models/clean_cnn_transformer/epoch_${i}.pth \
    ~/imagecaptioning-checkpoints/ \
    --zone=asia-southeast1-a
done

# Download vocabulary file
gcloud compute scp \
  instance-20251103-132126:/mnt/data/image-captioning-neural-networks/data/vocabulary.pkl \
  ~/imagecaptioning-checkpoints/ \
  --zone=asia-southeast1-a
```

### Method 3: Using rsync over SSH (Fastest)

```bash
# Create local directory
mkdir -p ~/imagecaptioning-checkpoints

# Download all files at once
gcloud compute scp \
  --recurse \
  instance-20251103-132126:/mnt/data/models/clean_cnn_transformer/*.pth \
  ~/imagecaptioning-checkpoints/ \
  --zone=asia-southeast1-a

# Download vocabulary
gcloud compute scp \
  instance-20251103-132126:/mnt/data/image-captioning-neural-networks/data/vocabulary.pkl \
  ~/imagecaptioning-checkpoints/ \
  --zone=asia-southeast1-a
```

## File Sizes

- Each checkpoint file: ~390MB
- Total for all 20 checkpoints: ~7.8GB
- Vocabulary file: ~2MB
- **Total download size: ~7.8GB**

## After Downloading

Once you've downloaded the files, place them in your project directory:

```
imagecaptioningCNN-transformer/
├── checkpoints/
│   ├── epoch_1.pth
│   ├── epoch_2.pth
│   ├── ...
│   └── epoch_20.pth
├── data/
│   └── vocabulary.pkl
├── model.py
├── generate_caption.py
└── test_all_20_epochs.py
```

Then you can run inference using:

```bash
python generate_caption.py --checkpoint checkpoints/epoch_20.pth --vocab data/vocabulary.pkl --image path/to/your/image.jpg
```

## Troubleshooting

**Issue: Permission denied**
- Solution: Make sure you're authenticated with `gcloud auth login`

**Issue: Connection timeout**
- Solution: Check that the GCP instance is running

**Issue: File not found**
- Solution: Verify the file paths by SSH-ing into the instance and running `ls -lh /mnt/data/models/clean_cnn_transformer/`

## Alternative: Upload to Google Drive

If you prefer to access the files via Google Drive, you can:

1. SSH into the GCP instance
2. Install rclone and configure Google Drive
3. Upload the files using:
   ```bash
   rclone copy /mnt/data/models/clean_cnn_transformer/ gdrive:ImageCaptioning/checkpoints/
   ```

This guide will be updated with Google Drive links once the files are uploaded.

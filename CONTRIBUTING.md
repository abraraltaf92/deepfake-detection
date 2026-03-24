# Contributing — Team Workflow Guide

This guide explains how our team collaborates using GitHub and Google Colab.

---

## Branch Structure

```
main (protected — requires PR + 1 approval)
├── dev/abrar
├── dev/sajan
├── dev/abdullah
└── dev/krishna
```

- **`main`** — stable, reviewed code only. No one pushes directly here.
- **`dev/<yourname>`** — your personal development branch. All your work happens here.

---

## Step-by-Step Workflow

### 1. One-Time Setup (First Time Only)

#### a) Create a GitHub Personal Access Token

1. Go to [GitHub Settings → Developer Settings → Personal Access Tokens → Tokens (classic)](https://github.com/settings/tokens)
2. Click **"Generate new token (classic)"**
3. Give it a name (e.g., `colab-access`)
4. Select scopes: `repo` (full control of private repositories)
5. Click **Generate token**
6. **Copy and save the token** — you won't see it again

#### b) Clone the Repo in Colab

Add this cell at the top of any notebook:

```python
# --- GitHub Setup (run once per Colab session) ---
import os

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone or pull the repo
REPO_URL = "https://<YOUR_TOKEN>@github.com/<username>/deepfake-detection.git"
REPO_DIR = "/content/drive/MyDrive/deepfake-detection"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull origin main

# Set your Git identity
!cd {REPO_DIR} && git config user.name "Your Name"
!cd {REPO_DIR} && git config user.email "your.email@example.com"
```

> ⚠️ **Never commit your token to the repo.** Replace `<YOUR_TOKEN>` each session or store it in a Colab secret.

#### c) Switch to Your Branch

```python
!cd {REPO_DIR} && git checkout dev/yourname
```

If your branch doesn't exist yet:

```python
!cd {REPO_DIR} && git checkout -b dev/yourname
!cd {REPO_DIR} && git push -u origin dev/yourname
```

---

### 2. Daily Workflow

#### Before Starting Work — Pull Latest Changes

```python
# Get latest from main into your branch
!cd {REPO_DIR} && git checkout dev/yourname
!cd {REPO_DIR} && git pull origin main
```

#### While Working

- Edit notebooks in Colab as usual
- Save your work to Google Drive normally

#### When Ready to Commit

**Option A: Commit from Colab (Recommended)**

```python
# Stage your changes
!cd {REPO_DIR} && git add notebooks/03_preprocessing.ipynb

# Commit with a descriptive message
!cd {REPO_DIR} && git commit -m "Add face detection with 20% margin cropping to preprocessing"

# Push to your branch
!cd {REPO_DIR} && git push origin dev/yourname
```

**Option B: Use Colab's Built-in GitHub Save**

1. Go to **File → Save a copy in GitHub**
2. Select the repository: `<username>/deepfake-detection`
3. Select your branch: `dev/yourname` — **NEVER select `main`**
4. **Edit the commit message** to describe what you changed
5. Update the file path to match the repo structure (e.g., `notebooks/02_eda.ipynb`)
6. Click **OK**

---

### 3. Opening a Pull Request

When your work is ready for review:

1. Go to the [GitHub repository](https://github.com/<username>/deepfake-detection)
2. You'll see a banner: *"dev/yourname had recent pushes"* → Click **"Compare & pull request"**
3. Set:
   - **Base:** `main`
   - **Compare:** `dev/yourname`
4. Write a title and description:
   ```
   Title: Add face detection preprocessing pipeline

   Description:
   - Implemented Haar cascade face detection with margin cropping
   - Added uniform frame sampling (16 frames per video)
   - Frames resized to 224x224 and saved as .jpg
   ```
5. **Assign a reviewer** (one of your teammates)
6. Click **"Create pull request"**

---

### 4. Reviewing a Pull Request

When a teammate assigns you as a reviewer:

1. Go to the PR on GitHub
2. Click the **"Files changed"** tab
3. Review the changes — look for:
   - Does the code run without errors?
   - Are paths using `configs/paths.py` instead of hardcoded values?
   - Is the commit message descriptive?
4. Leave comments if needed, or click **"Approve"**
5. The author then clicks **"Merge pull request"**

---

## Commit Message Guidelines

Write messages that describe **what you did and why**, not just "updated notebook."

**Good examples:**
```
Add Haar cascade face detection with 20% margin cropping
Fix data split leak — ensure same identity doesn't appear in train and test
Implement ResNet-18 baseline with class-weighted loss
Add ROC curve and confusion matrix to evaluation notebook
```

**Bad examples:**
```
Updated notebook
Fixed stuff
Changes
Final version
```

---

## File Naming Convention

Notebooks follow numbered order:
```
01_data_ingestion.ipynb
02_eda.ipynb
03_preprocessing.ipynb
04_model_baseline.ipynb
05_model_advanced.ipynb
06_evaluation.ipynb
```

Do **not** rename or renumber notebooks without team agreement.

---

## What NOT to Commit

These are excluded via `.gitignore`, but as a reminder:

- ❌ Dataset files (`.mp4`, frames, images) — stay on Google Drive
- ❌ Model weights (`.pth`, `.pt`, `.h5`) — stay on Google Drive
- ❌ Kaggle credentials (`kaggle.json`)
- ❌ `.ipynb_checkpoints/`
- ❌ Personal tokens or API keys

---

## Quick Reference

| Action | Command |
|--------|---------|
| Switch to your branch | `git checkout dev/yourname` |
| Pull latest main | `git pull origin main` |
| Stage a file | `git add notebooks/03_preprocessing.ipynb` |
| Commit | `git commit -m "Your descriptive message"` |
| Push | `git push origin dev/yourname` |
| Check status | `git status` |
| See what changed | `git diff` |

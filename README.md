# Early-Stage CKM Syndrome Risk Prediction Tool

## Overview
This is a Streamlit web application for predicting the risk of early-stage (Stage 1-2) CKM syndrome.

## Deployment Instructions (Streamlit Cloud)

1. **Upload to GitHub**
   - Create a new repository on GitHub.
   - Upload all files in this folder to the repository.
   - Ensure the structure is:
     ```
     /
     ├── app.py
     ├── requirements.txt
     ├── models/
     ├── data/
     └── plots/
     ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io/).
   - Log in with your GitHub account.
   - Click "New app".
   - Select your GitHub repository, branch (usually `main`), and main file path (`app.py`).
   - Click "Deploy".

3. **Wait for Installation**
   - Streamlit Cloud will install dependencies from `requirements.txt`.
   - Once finished, your app will be live!

## Local Run
To run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

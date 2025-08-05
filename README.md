# supreme-lamp

**Facial Detection & Recognition System with Milvus + MongoDB + Video Matching**

---

## 🧠 Project Overview

**`supreme-lamp`** is an intelligent facial recognition system built to process images and videos for facial detection, embedding generation, and identity matching. Using powerful models and a dual-database setup (Milvus for vector search, MongoDB for metadata), it supports both static image identification and video-based face matching.

---

## 🚀 Features

- 🔍 Face Detection & Embedding Extraction
- 🗂️ Vector Search with Milvus
- 📇 Metadata Storage with MongoDB
- 🖼️ Image-Based Person Search
- 🎥 Match Faces in Video (frame-by-frame)
- ⚙️ Central Config Management via `config.json`
- ✅ Unit-testable and modular

---

## 📁 Key Components

| File | Purpose |
|------|---------|
| `config.json` | Central config for model path, DB, thresholds |
| `face_model.py` | Loads and returns the face detection model |
| `data_storing.py` | Processes and stores facial embeddings & metadata |
| `retrieve_data.py` | Searches person info by face image |
| `video_matching.py` | Matches a person’s image against video frames |
| `test_folder/` | Sample reference image and test video |

---

# supreme-lamp
Supreme Lamp is categorical name for Facial Recognition System. Each individual image is processed and store in vector store and then it retrieve via single image or Video stream
🧠 Project Overview
supreme-lamp is a facial recognition system designed to process static images and live videos, detect faces, generate facial embeddings, and perform identity matching using Milvus vector search and MongoDB metadata. It supports real-time and batch processing workflows, making it suitable for surveillance, attendance systems, and security applications.

🚀 Core Features
🔍 Face Detection & Embedding Extraction
Uses a pre-configured face model to extract high-dimensional face embeddings from images and video frames.

🗂️ Data Storage and Search

Embeddings are stored in Milvus (vector database)

Associated metadata is stored in MongoDB

🖼️ Static Image Search
Match a new face against existing faces in the database using cosine similarity.

🎥 Video Matching
Match a person’s reference image against all frames of a video to locate appearances with visual overlay.

⚙️ Configurable Architecture
All settings (model paths, DB credentials, thresholds, frame skips, etc.) are managed in config.json.

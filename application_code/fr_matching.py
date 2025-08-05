import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
import os, json, base64
from face_model import prepare_model

# Initialize InsightFace model
# face_app = FaceAnalysis(name='buffalo_l')  # buffalo_l is a good model
# face_app.prepare(ctx_id=0)  # Use -1 for CPU, or set to 0 for GPU

# Load configuration from JSON
with open("model_config.json", "r") as f:
    data = json.load(f)


model = prepare_model(data)

# Step 1: Get reference embedding
def get_reference_embedding(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image from {image_path}")
        return
    # _, buffer = cv2.imencode(".jpg", image)
    # image_base64 = base64.b64encode(buffer)

    # person_image = str(image_base64)[2:-1]
    # # document["person_image"] = str(image_base64)[2:-1]
    # # Perform face detection
    faces = model.get(image)
    face_embeddings = faces[0].embedding.tolist()

    return face_embeddings

    # img = cv2.imread(image_path)
    # faces = model.get(img)
    # if not faces:
    #     raise ValueError("No face found in reference image.")
    # return faces[0].embedding

# Step 2: Process video
def match_faces_in_video(video_path, ref_embedding, threshold=0.38, output_dir='matched_frames', display=True):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    matched_frames = []

    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret = cap.grab()
        if not ret:
            break

        if frame_idx % data["frame_skips"] == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break

            print(f"Processing frame {frame_idx}")
            faces = model.get(frame)
            match_found = False

            for face in faces:
                bbox = face.bbox.astype(int)
                sim = np.dot(ref_embedding, face.embedding) / (
                    np.linalg.norm(ref_embedding) * np.linalg.norm(face.embedding)
                )

                if sim > threshold:
                    color = (10, 240, 0)  # Green box for match
                    match_found = True
                else:
                    color = (0, 0, 255)  # Red box for others

                # Draw bounding box and similarity
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, f'{sim:.2f}', (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if match_found:
                frame_path = f"{output_dir}/frame_{frame_idx}.jpg"
                cv2.imwrite(frame_path, frame)
                matched_frames.append((frame_idx, sim))
                print(f"--> Match found at frame {frame_idx} (similarity: {sim:.2f})")

            if display:
                cv2.imshow("Video Frame", frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    print("Interrupted by user.")
                    break

        frame_idx += 1

    cap.release()
    if display:
        cv2.destroyAllWindows()

    return matched_frames

if __name__ == "__main__":
    image_path = r"test_folder\reference_image.jpg"
    video_path = r"test_folder\test_1.mp4"
    ref_embedding = get_reference_embedding(image_path)
    matches = match_faces_in_video(video_path, ref_embedding)
    print(f"Found {len(matches)} matching frames.")
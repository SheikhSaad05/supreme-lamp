
import insightface
import json

# # Load configuration from JSON
# with open("config.json", "r") as f:
#     data = json.load(f)

def prepare_model(data):
    detction_threshold = data["detection_model"]["detection_threshold"]
    confidence_threshold = data["recognition_model"]["confidence_threshold"]
    model_type = data["model_type"]
    flag = data["everyface"]
    gpu_flag = data["gpu_flag"]
    frame_skips = data["frame_skips"]
    # Load face detection and recognition models
    model = insightface.app.FaceAnalysis(
        name=model_type,
        root=r"./insightface_models",
        allowed_modules=["detection", "recognition"]
    )
    model.prepare(ctx_id=gpu_flag, det_thresh=detction_threshold)
    return model



import os
import io
import torch
import librosa
import numpy as np
import joblib
import logging
import traceback
from flask import Flask, request, jsonify, render_template_string
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import hf_hub_download
import torch.nn as nn
from tempfile import NamedTemporaryFile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["HC", "MCI", "Dementia"]
REPO_ID = "gandalf513/memotagdementia"

# Load models on startup
try:
    logger.info(f"Loading models on device: {DEVICE}")
    
    # Load Whisper
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(DEVICE)
    whisper_model.eval()
    logger.info("Whisper model loaded successfully")

    # Load TF-IDF vectorizer
    vectorizer_path = hf_hub_download(repo_id=REPO_ID, filename="tfidf_vectorizer.pkl")
    tfidf_vectorizer = joblib.load(vectorizer_path)
    logger.info("TF-IDF vectorizer loaded successfully")

    # Define model
    class SimpleAudioTextClassifier(nn.Module):
        def __init__(self, text_feat_dim, acoustic_feat_dim=16, num_labels=3):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(text_feat_dim + acoustic_feat_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_labels)
            )

        def forward(self, text_feats, acoustic_feats):
            combined = torch.cat((text_feats, acoustic_feats), dim=1)
            return self.classifier(combined)

    # Load model
    model = SimpleAudioTextClassifier(text_feat_dim=256).to(DEVICE)
    model_path = hf_hub_download(repo_id=REPO_ID, filename="pytorch_model.bin")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    logger.info("Classification model loaded successfully")
    
except Exception as e:
    logger.error(f"Error during model loading: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Extract acoustic features - modified to handle both file paths and numpy arrays
def extract_acoustic_features(audio_input, sr=None):
    try:
        # Check if input is a file path or numpy array
        if isinstance(audio_input, str):
            y, sr = librosa.load(audio_input, sr=16000)
        else:
            y = audio_input
            if sr is None:
                sr = 16000
        
        # Extract features
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Handle potential NaN values for pitch
        try:
            pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))
            if np.isnan(pitch):
                pitch = 0.0
        except:
            pitch = 0.0
            
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        features = [duration, tempo, pitch] + mfcc_mean.tolist()
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    except Exception as e:
        logger.error(f"Error extracting acoustic features: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Web UI HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dementia Audio Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .container {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        #result {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            display: none;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #2980b9;
        }
        .error {
            color: red;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Dementia Audio Analysis</h1>
    <div class="container">
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="audioFile">Upload Audio File:</label>
                <input type="file" id="audioFile" name="file" accept="audio/*" required>
            </div>
            <button type="submit">Analyze</button>
        </form>
    </div>
    <div id="loading" style="display: none; text-align: center; margin-top: 20px;">
        <p>Analyzing... This may take up to 30 seconds</p>
    </div>
    <div id="result">
        <h2>Analysis Results</h2>
        <p><strong>Transcription:</strong> <span id="transcription"></span></p>
        <p><strong>Prediction:</strong> <span id="prediction"></span></p>
    </div>
    <div id="error" class="error" style="display: none; margin-top: 20px;">
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            
            const formData = new FormData();
            const fileInput = document.getElementById('audioFile');
            
            formData.append('file', fileInput.files[0]);
            
            fetch('/predict', {
                method: 'POST',
                body: formData,
                timeout: 60000  // Set a long timeout (60 seconds)
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Server error: ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                if (data.error) {
                    document.getElementById('error').textContent = 'Error: ' + data.error;
                    document.getElementById('error').style.display = 'block';
                } else {
                    document.getElementById('transcription').textContent = data.transcription;
                    document.getElementById('prediction').textContent = data.prediction;
                    document.getElementById('result').style.display = 'block';
                }
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('error').textContent = 'Error: ' + error.message;
                document.getElementById('error').style.display = 'block';
                console.error('Error:', error);
            });
        });
    </script>
</body>
</html>
'''

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            logger.error("No file uploaded")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if not file:
            logger.error("Empty file uploaded")
            return jsonify({"error": "Empty file uploaded"}), 400

        logger.info(f"Processing file: {file.filename}")
        
        # Use a temporary file that will be automatically deleted
        with NamedTemporaryFile(suffix='.' + file.filename.split('.')[-1], delete=False) as temp:
            file.save(temp.name)
            temp_path = temp.name

        try:
            # Process the audio file
            logger.info("Transcribing audio")
            audio, sr = librosa.load(temp_path, sr=16000)
            
            # Transcribe with Whisper
            input_features = whisper_processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(DEVICE)
            predicted_ids = whisper_model.generate(input_features)
            transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            logger.info(f"Transcription complete: {transcription[:50]}...")

            # TF-IDF transform
            text_feats = tfidf_vectorizer.transform([transcription]).toarray()
            text_feats_tensor = torch.tensor(text_feats, dtype=torch.float32).to(DEVICE)

            # Extract acoustic features
            logger.info("Extracting acoustic features")
            acoustic_feats = extract_acoustic_features(temp_path)

            # Make prediction
            logger.info("Making prediction")
            with torch.no_grad():
                outputs = model(text_feats_tensor, acoustic_feats)
                predicted_class = torch.argmax(outputs, dim=1).item()

            result = {
                "transcription": transcription,
                "prediction": LABELS[predicted_class]
            }
            logger.info(f"Prediction successful: {LABELS[predicted_class]}")
            
            return jsonify(result)

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Error processing audio: {str(e)}"}), 500
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"Temporary file removed: {temp_path}")

    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Health check endpoint for Render
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

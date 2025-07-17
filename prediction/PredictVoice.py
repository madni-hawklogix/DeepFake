from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import JsonResponse
from django.shortcuts import render
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from pydub import AudioSegment
import numpy as np
import os
from django.core.files.storage import FileSystemStorage
from datetime import datetime
from history.models import UserVoice

# Initialize model and feature extractor
MODEL_NAME = "motheecreator/Deepfake-audio-detection"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model and feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

def load_audio_as_np(file_path, target_sr=16000):
    # Load with pydub (auto-detects format)
    audio = AudioSegment.from_file(file_path)
    # Convert to mono and target sample rate
    audio = audio.set_channels(1).set_frame_rate(target_sr)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)
    return samples, target_sr

def process_audio(audio_file):
    samples, sample_rate = load_audio_as_np(audio_file)
    inputs = feature_extractor(samples, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).squeeze()
        
    return probabilities.cpu().numpy()

@csrf_exempt
def predictVoice(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return render(request, 'voice.html', {'error': 'No file uploaded'})

        # Save the original uploaded audio
        fs = FileSystemStorage()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S').replace(':','-')
        original_file_name = f"original_{uploaded_file.name.split('.')[0]}-{timestamp}.wav"
        original_file_path = os.path.join(settings.BASE_DIR / 'uploaded_voices', original_file_name)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(original_file_path), exist_ok=True)
        
        # Save the file
        with open(original_file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        # Process the audio file
        try:
            probabilities = process_audio(original_file_path)
            real_prob = probabilities[1]  # Human (Real)
            fake_prob = probabilities[0]  # AI (Fake)

            print(real_prob*100, fake_prob*100)
            
            prediction = "real" if real_prob > fake_prob else "fake"
            confidence = (real_prob - fake_prob) * 100
            print(confidence)
            
            # Generate description based on prediction
            if prediction == "fake":
                description = (
                    f"Detected as AI-generated audio with confidence {confidence:.1f}%. "
                    "The model identified patterns typical of synthetic voices, such as unnatural prosody, lack of background noise, or digital artifacts."
                )
            else:
                description = (
                    f"Detected as human audio with confidence {confidence:.1f}%. "
                    "The model found natural speech patterns, background noise, and intonation consistent with real human recordings."
                )
            
            # Save to database
            user_voice = UserVoice.objects.create(
                user=request.user,
                original_voice=original_file_name,
                result=prediction,
                fake_prediction = float(f"{fake_prob * 100:.1f}"),
                real_prediction = float(f"{real_prob * 100:.1f}"),
            )
            
            context = {
                'prediction': prediction,
                'confidences': {
                    'real': float(f"{real_prob * 100:.1f}"),
                    'fake': float(f"{fake_prob * 100:.1f}")
                },
                'description': description,
                'original_voice_url': fs.url(original_file_name)
            }
            
            return render(request, 'voice.html', context)
            
        except Exception as e:
            return render(request, 'voice.html', {'error': f'Error processing audio: {str(e)}'})
    
    return render(request, 'voice.html', {'error': 'Invalid request method'})

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
import base64
from io import BytesIO

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

# def load_audio_from_base64(base64_str, target_sr=16000):
#     audio_bytes = base64.b64decode(base64_str)
#     print("Audio Bytes:  ",audio_bytes)
#     print("Debug 2")
#     audio = AudioSegment.from_file(BytesIO(audio_bytes))
#     print("Debug 3")
#     audio = audio.set_channels(1).set_frame_rate(target_sr)
#     print("Debug 4")
#     samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)
#     print("Debug 5")
#     return samples, target_sr

def load_audio_from_base64(base64_str, target_sr=16000, channels=1, bit_depth=16):
    try:
        if base64_str.startswith('data:audio'):
            base64_str = base64_str.split(',')[1]
        audio_bytes = base64.b64decode(base64_str)

        sample_width = bit_depth // 8

        if len(audio_bytes) % sample_width != 0:
            padding_size = sample_width - (len(audio_bytes) % sample_width)
            audio_bytes += b'\0' * padding_size

        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

        audio_data = audio_data.astype(np.float32) / (2 ** 15)
        if target_sr != 44100:
            resample_factor = target_sr / 44100
            resampled_audio = np.interp(
                np.linspace(0, len(audio_data), int(len(audio_data) * resample_factor)),
                np.arange(len(audio_data)),
                audio_data
            )
            audio_data = resampled_audio
        return audio_data, target_sr
    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None

# def process_audio(audio_file):
#     samples, sample_rate = load_audio_as_np(audio_file)
#     inputs = feature_extractor(samples, sampling_rate=sample_rate, return_tensors="pt")
#     inputs = {k: v.to(device) for k, v in inputs.items()}
    
#     with torch.no_grad():
#         logits = model(**inputs).logits
#         probabilities = torch.softmax(logits, dim=-1).squeeze()
        
#     return probabilities.cpu().numpy()

@csrf_exempt
def predictVoice(request):
    if request.method == 'POST':
        import json
        try:
            data = json.loads(request.body)
            base64_audio = data.get('audio_base64')
        except Exception:
            base64_audio = None
        if not base64_audio:
            return JsonResponse({'error': 'No audio data provided'}, status=400)
        try:
            samples, sample_rate = load_audio_from_base64(base64_audio)
            inputs = feature_extractor(samples, sampling_rate=sample_rate, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1).squeeze()
            real_prob = probabilities[1]
            fake_prob = probabilities[0]
            prediction = "real" if real_prob > fake_prob else "fake"
            confidence = (real_prob - fake_prob) * 100
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
                original_voice="something",
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
            }
            print("context")
            # return render(request, 'voice.html', context)
            return JsonResponse(context)
            
        except Exception as e:
            return render(request, 'voice.html', {'error': f'Error processing audio: {str(e)}'})
    
    return render(request, 'voice.html', {'error': 'Invalid request method'})


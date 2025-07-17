from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import JsonResponse
from django.shortcuts import render
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import warnings
import base64
import io
from history.models import UserImage
import os
from django.core.files.storage import FileSystemStorage
from datetime import datetime

warnings.filterwarnings("ignore")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Using BASE_DIR from settings
BASE_DIR = settings.BASE_DIR



def numpy_image_to_base64(image_array):
    # Ensure the image is in a format suitable for encoding
    success, buffer = cv2.imencode('.jpg', image_array)

    if success:
        # Encode the byte array to Base64
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image
    else:
        raise ValueError("Image encoding failed.")
    



mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load(BASE_DIR / "models/resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()


# def predictImage(request):
#     if request.method == 'POST':
#         uploaded_file = request.FILES.get('file')
#         if not uploaded_file:
#             return JsonResponse({'error': 'No file uploaded'})

#         image = Image.open(uploaded_file)

#         face = mtcnn(image)
#         if face is None:
#             return JsonResponse({'error': 'No face detected'})

#         face = face.unsqueeze(0)
#         face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)


#         # convert the face into a numpy array to be able to plot it
#         prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
#         prev_face = prev_face.astype('uint8')
        
#         face = face.to(DEVICE)
#         face = face.to(torch.float32)
#         face = face / 255.0
#         face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

#         target_layers = [model.block8.branch1[-1]]
        
#         # Correct usage
#         cam = GradCAM(model=model, target_layers=target_layers)
        
#         targets = [ClassifierOutputTarget(0)]
        
#         grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
#         grayscale_cam = grayscale_cam[0, :]
#         visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
#         face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)
#         face_with_mask_base64 = numpy_image_to_base64(face_with_mask)
#         face_with_mask_base64 = f"data:image/png;base64, {face_with_mask_base64}"
        
#         with torch.no_grad():
#             output = torch.sigmoid(model(face).squeeze(0))
#             prediction = "real" if output.item() < 0.5 else "fake"
#             real_prediction = 1 - output.item()
#             fake_prediction = output.item()
            
#             confidences = {
#                 'real': round(real_prediction * 100),
#                 'fake': round(fake_prediction * 100)
#             }
        
#         context = {'prediction': prediction, 'confidences': confidences,'face_with_mask':face_with_mask_base64}
#         return render(request, 'image.html',context)
#     return JsonResponse({'error': 'Invalid request method'})

@csrf_exempt
def predictImage(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return render(request, 'image.html', {'error': 'No file uploaded'})

        # Open image and convert to RGB mode
        image = Image.open(uploaded_file).convert('RGB')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S').replace(':','-')

        # Save the original uploaded image
        fs = FileSystemStorage()
        original_file_name = f"original_{uploaded_file.name.split('.')[0]}-{timestamp}.png"
        original_file_path = os.path.join(settings.BASE_DIR / 'uploaded_images', original_file_name)
        image.save(original_file_path)  # Save the original image
        original_file_url = fs.url(original_file_name)  # URL of the original image

        # Face detection and processing
        face = mtcnn(image)
        if face is None:
            return render(request, 'image.html', {'error': 'No face detected'})

        face = face.unsqueeze(0)
        face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

        # Convert the face into a numpy array to be able to plot it
        prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
        prev_face = prev_face.astype('uint8')

        face = face.to(DEVICE)
        face = face.to(torch.float32)
        face = face / 255.0
        face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

        target_layers = [model.block8.branch1[-1]]
        
        # Correct usage
        cam = GradCAM(model=model, target_layers=target_layers)
        
        targets = [ClassifierOutputTarget(0)]
        
        grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
        face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

        # Save the processed image (with mask)
        processed_file_name = f"processed_{uploaded_file.name.split('.')[0]}-{timestamp}.png"
        processed_file_path = os.path.join(settings.BASE_DIR / 'uploaded_images', processed_file_name)
        cv2.imwrite(processed_file_path, face_with_mask)  # Save the processed image
        processed_file_url = fs.url(processed_file_name)  # URL of the processed image

        # Convert processed image to base64 for rendering
        face_with_mask_base64 = numpy_image_to_base64(face_with_mask)
        face_with_mask_base64 = processed_file_name

        with torch.no_grad():
            output = torch.sigmoid(model(face).squeeze(0))
            prediction = "real" if output.item() < 0.5 else "fake"
            real_prediction = 1 - output.item()
            fake_prediction = output.item()
            
            confidences = {
                'real': round(real_prediction * 100),
                'fake': round(fake_prediction * 100)
            }

        user_image = UserImage.objects.create(
            user=request.user,
            original_image= original_file_name,
            processed_image= processed_file_name,
            result =  prediction,
            fake_prediction =  confidences['fake'],
            real_prediction =  confidences['real']
        )
        
        context = {
            'prediction': prediction,
            'confidences': confidences,
            'face_with_mask': face_with_mask_base64,
            'original_image_url': original_file_url,  # URL of the original image
            'processed_image_url': processed_file_url  # URL of the processed image
        }
        
        return render(request, 'image.html', context)    
    
    return render(request, 'image.html', {'error': 'Invalid request method'})

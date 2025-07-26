import os
import torch
from flask import Flask, render_template, request, jsonify
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
from torchvision.models import efficientnet_b0

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_b0(pretrained=True)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)  # Adjust to 4 output classes
model.load_state_dict(torch.load('cotton_cure_model.pth'))
model.to(device)
model.eval()

# Transformation for incoming image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Define classes for mapping predictions
class_names = ['Bacterial Blight', 'Healthy', 'Powdery Mildew', 'Target Spot']

# API URL for Gemini (example placeholder)
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent'  # Replace with actual Gemini API URL

# Function to call Gemini API

# Function to call Gemini API with an API key
def get_disease_info(disease_name):


    if disease_name == "Healthy":
        # Predefined information for "Healthy"
        return {
            "info": "The plant is healthy and shows no signs of disease.",
            "diagnosis": "No diagnosis is required as the plant is healthy.",
            "prevention": "Ensure regular monitoring and optimal farming practices to maintain plant health.",
        }

        # For other diseases, use the API


    api_key = 'YOUR_API_KEY'  # Replace with your actual API key
    url = f"{GEMINI_API_URL}?key={api_key}"
    
    # JSON body for the POST request
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Write a concise explanation about {disease_name} in cotton plants in exactly three paragraphs. The first paragraph should briefly describe the disease, its symptoms, and how it affects cotton plants. The second paragraph should explain how to diagnose it add some pesticides or medicines. The third paragraph should detail preventive measures to control or manage the disease."
                    }
                ]
            }
        ]
    }

    headers = {
        'Content-Type': 'application/json',
    }

    try:
        # Make the POST request
        response = requests.post(url, json=payload, headers=headers)
        print("Response Status:", response.status_code)
        print("Response Body:", response.text)

        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Extract text from the nested structure
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )

            # Return the text as-is for simplicity (since it's structured into three paragraphs)
            return {
                "info": text.split("\n\n")[0] if "\n\n" in text else text,
                "diagnosis": text.split("\n\n")[1] if len(text.split("\n\n")) > 1 else "Diagnosis information unavailable.",
                "prevention": text.split("\n\n")[2] if len(text.split("\n\n")) > 2 else "Prevention steps are not available.",
            }
        else:
            return {
                "info": "Unable to fetch disease info from Gemini API.",
                "diagnosis": "N/A",
                "prevention": "N/A",
            }
    except Exception as e:
        print("Exception occurred:", e)
        return {
            "info": "An error occurred while fetching disease info.",
            "diagnosis": "N/A",
            "prevention": "N/A",
        }


# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read the image file and process it
    image = Image.open(file.stream).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Predict the disease class
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    # Get additional info from Gemini API
    disease_info = get_disease_info(predicted_class)

    # Return the parsed information as JSON
    result = {
        'disease': predicted_class,
        'info': disease_info.get('info', 'Information not available for this disease.'),
        'diagnosis': disease_info.get('diagnosis', 'Diagnosis information is unavailable.'),
        'prevention': disease_info.get('prevention', 'Prevention steps are not available.'),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)

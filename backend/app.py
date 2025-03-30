from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import PyPDF2
import requests  # For API calls

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Replace with your Gemini API key and endpoint
GEMINI_API_KEY = "AIzaSyAmhpZIMQq07zy-sG8DyKYexGTQ1xmsjaY"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

def gemini_summarize(text):
    try:
        response = requests.post(
            GEMINI_API_URL,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY
            },
            json={
                "contents": [{
                    "parts": [{
                        "text": f"Please provide a concise summary of the following text: {text}"
                    }]
                }]
            }
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error summarizing: {str(e)}"

def gemini_flashcards(text):
    try:
        response = requests.post(
            GEMINI_API_URL,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY
            },
            json={
                "contents": [{
                    "parts": [{
                        "text": f"Create 2 flashcards from this text. Format each as 'Q: [question] A: [answer]': {text}"
                    }]
                }]
            }
        )
        response.raise_for_status()
        response_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        # Parse the response into flashcards
        flashcards = []
        for line in response_text.split('\n'):
            if line.startswith('Q:'):
                question = line[2:].strip()
                answer = next((line[2:].strip() for line in response_text.split('\n') if line.startswith('A:')), '')
                flashcards.append({"front": question, "back": answer})
        return flashcards
    except Exception as e:
        return [{"front": "Error", "back": str(e)}]

def gemini_chat(message):
    try:
        response = requests.post(
            GEMINI_API_URL,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY
            },
            json={
                "contents": [{
                    "parts": [{
                        "text": message
                    }]
                }]
            }
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error chatting: {str(e)}"

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''
            return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

@app.route('/upload_pdf_summary', methods=['POST'])
def upload_pdf_summary():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    text = extract_text_from_pdf(file_path)
    summary = gemini_summarize(text)
    os.remove(file_path)  # Clean up
    return jsonify({"summary": summary})

@app.route('/upload_pdf_flashcards', methods=['POST'])
def upload_pdf_flashcards():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    text = extract_text_from_pdf(file_path)
    flashcards = gemini_flashcards(text)
    os.remove(file_path)  # Clean up
    return jsonify({"flashcards": flashcards})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '').strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400
    response = gemini_chat(message)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
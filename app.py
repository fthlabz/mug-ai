from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    api_key = data.get('token')
    prompt = data.get('prompt')
    negative_prompt = data.get('negative_prompt', "")
    width = data.get('width', 1344)
    height = data.get('height', 768)
    # Modelin kalite ve sadakat ayarları
    guidance_scale = data.get('guidance_scale', 8.0)
    num_inference_steps = data.get('num_inference_steps', 50)

    headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
    
    # SDXL Modeli
    payload = {
        "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        "input": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "refine": "expert_ensemble_refiner",
            "apply_watermark": False
        }
    }
    
    try:
        res = requests.post("https://api.replicate.com/v1/predictions", headers=headers, json=payload)
        if res.status_code != 201:
            return jsonify({"error": "Replicate API hatası"}), 500
        return jsonify(res.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status/<prediction_id>', methods=['POST'])
def status(prediction_id):
    api_key = request.json.get('token')
    headers = {"Authorization": f"Token {api_key}"}
    try:
        res = requests.get(f"https://api.replicate.com/v1/predictions/{prediction_id}", headers=headers)
        return jsonify(res.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cancel', methods=['POST'])
def cancel():
    data = request.json
    api_key = data.get('token')
    prediction_id = data.get('id')
    if not api_key or not prediction_id:
        return jsonify({"status": "ignored"}), 200
        
    headers = {"Authorization": f"Token {api_key}"}
    try:
        requests.post(f"https://api.replicate.com/v1/predictions/{prediction_id}/cancel", headers=headers)
        return jsonify({"status": "cancelled"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

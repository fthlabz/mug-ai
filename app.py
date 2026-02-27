from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests

app = Flask(__name__)
# Hangi kapıdan gelirse gelsin kabul et (CORS İzni)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def index():
    # Arayüzü render et
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    api_key = data.get('token')
    
    # Frontend'den gelen dinamik "Master Prompt" ve "Negative Prompt"
    prompt = data.get('prompt')
    negative_prompt = data.get('negative_prompt', "")
    
    # Boyutlar (Wrap formatı için 1344x768)
    width = data.get('width', 1344)
    height = data.get('height', 768)
    
    # Senin belirlediğin Pro ayarlamalar ve Otonom Seed
    guidance_scale = data.get('guidance_scale', 6.5)
    num_inference_steps = data.get('num_inference_steps', 50)
    seed = data.get('seed')

    if not api_key or not prompt:
        return jsonify({"error": "Eksik bilgi gönderildi."}), 400

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }

    # Replicate'e gönderilecek temel input verisi
    input_data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "refine": "expert_ensemble_refiner",
        "apply_watermark": False
    }
    
    # Eğer seed geldiyse input'a ekle (Tekil benzersiz üretim için kritik nokta)
    if seed is not None:
        input_data["seed"] = seed

    # Replicate SDXL Modeli Payload'ı
    payload = {
        "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        "input": input_data
    }

    try:
        # Replicate'e işlemi başlatma emri
        res = requests.post("https://api.replicate.com/v1/predictions", headers=headers, json=payload)
        if res.status_code != 201:
            return jsonify({"error": "Replicate API hatası", "details": res.json()}), 500
            
        return jsonify(res.json())
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status/<prediction_id>', methods=['POST'])
def status(prediction_id):
    # Arayüz bu adrese sorarak resmin bitip bitmediğini kontrol eder
    api_key = request.json.get('token')
    headers = {"Authorization": f"Token {api_key}"}
    
    try:
        res = requests.get(f"https://api.replicate.com/v1/predictions/{prediction_id}", headers=headers)
        return jsonify(res.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cancel', methods=['POST'])
def cancel():
    # Sayfa kapanırsa "Acil Fren" buraya gelir ve üretimi durdurarak bakiyeni korur
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
    # Gunicorn ile Render üzerinde çalışması için port 5000 ayarlı
    app.run(debug=True, host='0.0.0.0', port=5000)

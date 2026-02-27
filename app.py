from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import time

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
    prompt = data.get('prompt')
    width = data.get('width', 1024)
    height = data.get('height', 1024)

    if not api_key or not prompt:
        return jsonify({"error": "Eksik bilgi gönderildi."}), 400

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }

    # Replicate SDXL Modeli
    payload = {
        "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        "input": {
            "prompt": prompt,
            "width": width,
            "height": height,
            "refine": "expert_ensemble_refiner",
            "apply_watermark": False
        }
    }

    try:
        # Replicate'e işlemi başlatma emri
        start_res = requests.post("https://api.replicate.com/v1/predictions", headers=headers, json=payload)
        if start_res.status_code != 201:
            return jsonify({"error": "Replicate API hatası."}), 500

        prediction = start_res.json()
        get_url = prediction['urls']['get']

        # Sonuç Bekleme Döngüsü (Backend tarafında polling)
        while True:
            time.sleep(2) # API'yi yormamak için 2 saniye bekle
            poll_res = requests.get(get_url, headers=headers).json()
            
            if poll_res['status'] == 'succeeded':
                return jsonify({"output": poll_res['output'][0]})
            elif poll_res['status'] == 'failed':
                return jsonify({"error": "Görsel üretimi başarısız oldu."}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

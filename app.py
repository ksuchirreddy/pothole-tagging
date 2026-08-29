"""
AI Pothole Tagging System - Flask Backend
Computer Vision based pothole detection using OpenCV
"""
import os
import cv2
import numpy as np
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import base64
import io

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')

if os.path.exists(os.path.join(BASE_DIR, 'demo-data')):
    DEMO_DATA_FOLDER = os.path.join(BASE_DIR, 'demo-data')
elif os.path.exists(os.path.join(BASE_DIR, '../demo-data')):
    DEMO_DATA_FOLDER = os.path.join(BASE_DIR, '../demo-data')
else:
    DEMO_DATA_FOLDER = os.path.join(BASE_DIR, 'demo-data')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(DEMO_DATA_FOLDER, exist_ok=True)

# In-memory storage for demo
pothole_database = []

class PotholeDetector:
    """Pothole detection using computer vision techniques"""

    def __init__(self):
        self.min_area = 500
        self.max_area = 50000

    def preprocess_image(self, image):
        """Preprocess image for pothole detection"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        return enhanced

    def detect_edges(self, image):
        """Detect edges using Canny edge detection"""
        # Adaptive threshold for edge detection
        edges = cv2.Canny(image, 50, 150)

        # Morphological operations to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

        return edges

    def detect_potholes_contour(self, image):
        """Detect potholes using contour analysis"""
        processed = self.preprocess_image(image)
        edges = self.detect_edges(processed)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        potholes = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if self.min_area < area < self.max_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate shape properties
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                # Approximate contour to polygon
                epsilon = 0.02 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Calculate severity based on area and depth estimation
                severity = self.estimate_severity(area, circularity, image[y:y+h, x:x+w])

                # Calculate confidence based on shape properties
                confidence = self.calculate_confidence(circularity, len(approx), area)

                if confidence > 0.3:
                    potholes.append({
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'area': float(area),
                        'circularity': float(circularity),
                        'severity': severity,
                        'confidence': float(confidence),
                        'contour_points': contour.tolist()
                    })

        return potholes

    def estimate_severity(self, area, circularity, roi):
        """Estimate pothole severity based on visual features"""
        # Base severity on area
        if area < 1000:
            base_severity = "Low"
        elif area < 5000:
            base_severity = "Medium"
        elif area < 15000:
            base_severity = "High"
        else:
            base_severity = "Critical"

        # Adjust based on circularity (more circular = more likely pothole)
        if circularity > 0.6:
            severity_map = {"Low": "Low", "Medium": "Medium", "High": "High", "Critical": "Critical"}
        else:
            severity_map = {"Low": "Low", "Medium": "Low", "High": "Medium", "Critical": "High"}

        return severity_map.get(base_severity, "Medium")

    def calculate_confidence(self, circularity, vertices, area):
        """Calculate detection confidence"""
        confidence = 0.0

        # Circularity factor (potholes tend to be somewhat circular/oval)
        confidence += min(circularity * 0.5, 0.3)

        # Vertex count factor (potholes typically have 4-12 vertices)
        if 4 <= vertices <= 12:
            confidence += 0.2
        elif vertices > 12:
            confidence += 0.1

        # Area factor
        if 500 < area < 20000:
            confidence += 0.3
        elif area < 50000:
            confidence += 0.2

        return min(confidence, 1.0)

    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image"""
        result = image.copy()

        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']
            severity = det['severity']
            confidence = det['confidence']

            # Color based on severity
            colors = {
                'Low': (0, 255, 0),      # Green
                'Medium': (0, 255, 255),  # Yellow
                'High': (0, 165, 255),    # Orange
                'Critical': (0, 0, 255)   # Red
            }
            color = colors.get(severity, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)

            # Draw label background
            label = f"Pothole #{i+1}: {severity} ({confidence:.0%})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result, (x, y - label_size[1] - 10), (x + label_size[0] + 10, y), color, -1)

            # Draw label text
            cv2.putText(result, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Draw contour
            if 'contour_points' in det:
                contour = np.array(det['contour_points'], dtype=np.int32)
                cv2.drawContours(result, [contour], -1, color, 2)

        return result


detector = PotholeDetector()

@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'AI Pothole Tagging System',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/detect', methods=['POST'])
def detect_potholes():
    """Detect potholes in uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Detect potholes
        detections = detector.detect_potholes_contour(image)

        # Draw detections on image
        result_image = detector.draw_detections(image, detections)

        # Save result image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_filename = f'detection_{timestamp}.jpg'
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        cv2.imwrite(result_path, result_image)

        # Convert result to base64 for frontend
        _, buffer = cv2.imencode('.jpg', result_image)
        result_base64 = base64.b64encode(buffer).decode('utf-8')

        # Save to database
        detection_record = {
            'id': len(pothole_database) + 1,
            'timestamp': datetime.now().isoformat(),
            'filename': file.filename,
            'detections': detections,
            'total_count': len(detections),
            'result_image': result_filename
        }
        pothole_database.append(detection_record)

        return jsonify({
            'success': True,
            'detections': detections,
            'total_count': len(detections),
            'result_image': result_base64,
            'record_id': detection_record['id']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect/base64', methods=['POST'])
def detect_potholes_base64():
    """Detect potholes from base64 encoded image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No base64 image provided'}), 400

        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[-1])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Detect potholes
        detections = detector.detect_potholes_contour(image)

        # Draw detections
        result_image = detector.draw_detections(image, detections)

        # Convert to base64
        _, buffer = cv2.imencode('.jpg', result_image)
        result_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'detections': detections,
            'total_count': len(detections),
            'result_image': result_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get detection history"""
    return jsonify({
        'success': True,
        'history': pothole_database
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get detection statistics"""
    if not pothole_database:
        return jsonify({
            'total_detections': 0,
            'total_images': 0,
            'severity_distribution': {},
            'recent_activity': []
        })

    total_detections = sum(record['total_count'] for record in pothole_database)

    severity_counts = {'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0}
    for record in pothole_database:
        for det in record['detections']:
            severity_counts[det['severity']] = severity_counts.get(det['severity'], 0) + 1

    return jsonify({
        'total_detections': total_detections,
        'total_images': len(pothole_database),
        'severity_distribution': severity_counts,
        'recent_activity': pothole_database[-10:]
    })

@app.route('/api/demo-images', methods=['GET'])
def get_demo_images():
    """Get list of demo images"""
    demo_images = []
    if os.path.exists(DEMO_DATA_FOLDER):
        for fname in os.listdir(DEMO_DATA_FOLDER):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                demo_images.append(fname)
    return jsonify({'images': demo_images})

@app.route('/api/demo/<filename>', methods=['GET'])
def get_demo_image(filename):
    """Serve demo image"""
    filepath = os.path.join(DEMO_DATA_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return jsonify({'error': 'Image not found'}), 404

@app.route('/api/result/<filename>', methods=['GET'])
def get_result_image(filename):
    """Serve result image"""
    filepath = os.path.join(RESULTS_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return jsonify({'error': 'Result not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("Starting AI Pothole Tagging System Backend...")
    print(f"Backend running on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
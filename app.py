import os
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
import time

# Import key components from the original application
from main import RingVisualizerApp

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
RINGS_FOLDER = 'rings'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create required folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(RINGS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['RINGS_FOLDER'] = RINGS_FOLDER

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class RingProcessorService:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        
        # Initialize finger mapping (same as in RingVisualizerApp)
        self.finger_map = {
            "Thumb": (mp.solutions.hands.HandLandmark.THUMB_CMC, mp.solutions.hands.HandLandmark.THUMB_MCP),
            "Index": (mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP, mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP),
            "Middle": (mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP),
            "Ring": (mp.solutions.hands.HandLandmark.RING_FINGER_MCP, mp.solutions.hands.HandLandmark.RING_FINGER_PIP),
            "Pinky": (mp.solutions.hands.HandLandmark.PINKY_MCP, mp.solutions.hands.HandLandmark.PINKY_PIP)
        }
        
        # Load available rings
        self.load_available_rings()
    
    def load_available_rings(self):
        """Load all available ring images from the rings folder"""
        self.available_rings = {}
        for filename in os.listdir(app.config['RINGS_FOLDER']):
            if allowed_file(filename):
                ring_id = os.path.splitext(filename)[0]
                ring_path = os.path.join(app.config['RINGS_FOLDER'], filename)
                self.available_rings[ring_id] = ring_path
    
    def overlay_image_alpha(self, background, overlay, x, y, alpha_mask):
        """
        Overlay `overlay` onto `background` at position (x, y) with an alpha mask.
        """
        h, w = overlay.shape[:2]
        
        # Check boundaries
        if x < 0:
            overlay = overlay[:, -x:]
            alpha_mask = alpha_mask[:, -x:]
            w += x
            x = 0
        if y < 0:
            overlay = overlay[-y:, :]
            alpha_mask = alpha_mask[-y:, :]
            h += y
            y = 0
        if x + w > background.shape[1]:
            overlay = overlay[:, :background.shape[1]-x]
            alpha_mask = alpha_mask[:, :background.shape[1]-x]
            w = background.shape[1] - x
        if y + h > background.shape[0]:
            overlay = overlay[:background.shape[0]-y, :]
            alpha_mask = alpha_mask[:background.shape[0]-y, :]
            h = background.shape[0] - y
        
        if w <= 0 or h <= 0:
            return
        
        # Get ROI
        roi = background[y:y+h, x:x+w]
        
        # Apply alpha blending
        for c in range(3):
            roi[:, :, c] = (alpha_mask / 255.0 * overlay[:, :, c] + 
                            (1.0 - alpha_mask / 255.0) * roi[:, :, c])
        
        # Update background
        background[y:y+h, x:x+w] = roi
    
    def process_image(self, hand_image_path, ring_image_path, params):
        """Process the image with the given parameters"""
        try:
            # Extract parameters with defaults
            finger = params.get('finger', 'Middle')
            ring_width = int(params.get('ring_width', 100))
            ring_height = int(params.get('ring_height', 50))
            x_offset = int(params.get('x_offset', 0))
            y_offset = int(params.get('y_offset', 0))
            rotation = float(params.get('rotation', 0))  # New parameter for rotation
            
            # Load the hand image
            hand_image = cv2.imread(hand_image_path)
            if hand_image is None:
                return None, "Failed to load the hand image."
            
            # Load the ring image with alpha channel
            ring_image = cv2.imread(ring_image_path, cv2.IMREAD_UNCHANGED)
            if ring_image is None:
                return None, "Failed to load the ring image."
            
            # Resize the ring image based on parameters
            ring_image = cv2.resize(ring_image, (ring_width, ring_height))
            
            # Apply rotation if needed
            if rotation != 0:
                # Get the center of the image
                center = (ring_width // 2, ring_height // 2)
                # Get the rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
                # Apply the rotation
                ring_image = cv2.warpAffine(ring_image, rotation_matrix, (ring_width, ring_height), 
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
            
            # Get selected finger landmarks
            if finger not in self.finger_map:
                return None, f"Invalid finger selection: {finger}"
            landmark_pair = self.finger_map[finger]
            
            # Process with MediaPipe
            with self.mp_hands.Hands(static_image_mode=True) as hands:
                hand_image_rgb = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
                results = hands.process(hand_image_rgb)
                
                image_height, image_width = hand_image.shape[:2]
                result_image = hand_image.copy()
                
                if not results.multi_hand_landmarks:
                    return None, "No hand detected in the image."
                    
                for landmarks in results.multi_hand_landmarks:
                    # Get selected finger landmarks
                    mcp = landmarks.landmark[landmark_pair[0]]
                    pip = landmarks.landmark[landmark_pair[1]]
                    
                    # Convert to pixel coordinates
                    mcp_x, mcp_y = int(mcp.x * image_width), int(mcp.y * image_height)
                    pip_x, pip_y = int(pip.x * image_width), int(pip.y * image_height)
                    
                    # Calculate midpoint for ring placement
                    mid_y = (mcp_y + pip_y) // 2
                    
                    # Position the ring on the finger with offsets
                    self.overlay_image_alpha(
                        result_image,
                        ring_image[:, :, :3],
                        mcp_x - ring_width//2 + x_offset,  # Center with x offset
                        mid_y - ring_height//2 + y_offset,  # Center vertically with y offset
                        ring_image[:, :, 3]
                    )
                    
                    # Only process the first detected hand
                    break
            
            return result_image, None
            
        except Exception as e:
            return None, str(e)

# Initialize the ring processor service
ring_processor = RingProcessorService()

# API endpoint to get available rings
@app.route('/api/rings', methods=['GET'])
def get_rings():
    ring_processor.load_available_rings()  # Refresh ring list
    rings = [{"id": ring_id, "name": ring_id.replace("_", " ").title()} 
             for ring_id in ring_processor.available_rings.keys()]
    return jsonify({"rings": rings})

# API endpoint to upload a ring
@app.route('/api/rings/upload', methods=['POST'])
def upload_ring():
    if 'ring' not in request.files:
        return jsonify({"error": "No ring file provided"}), 400
    
    file = request.files['ring']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        ring_id = request.form.get('ring_id', str(uuid.uuid4()))
        filename = secure_filename(f"{ring_id}.png")
        filepath = os.path.join(app.config['RINGS_FOLDER'], filename)
        file.save(filepath)
        
        # Refresh available rings
        ring_processor.load_available_rings()
        
        return jsonify({
            "message": "Ring uploaded successfully",
            "ring_id": ring_id
        })
    
    return jsonify({"error": "Invalid file type"}), 400

# API endpoint for hand detection analysis
@app.route('/api/analyze', methods=['POST'])
def analyze_hand():
    if 'hand_image' not in request.files:
        return jsonify({"error": "No hand image provided"}), 400
    
    file = request.files['hand_image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded hand image
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze hand using MediaPipe
        with mp.solutions.hands.Hands(static_image_mode=True) as hands:
            image = cv2.imread(filepath)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            if not results.multi_hand_landmarks:
                return jsonify({
                    "error": "No hand detected in the image", 
                    "image_path": filepath
                }), 400
            
            # Extract hand landmark information for all fingers
            image_height, image_width = image.shape[:2]
            finger_positions = {}
            
            for finger_name, (landmark1, landmark2) in ring_processor.finger_map.items():
                landmarks = results.multi_hand_landmarks[0]  # Use first detected hand
                
                # Get landmark positions
                lm1 = landmarks.landmark[landmark1]
                lm2 = landmarks.landmark[landmark2]
                
                # Convert to pixel coordinates
                x1, y1 = int(lm1.x * image_width), int(lm1.y * image_height)
                x2, y2 = int(lm2.x * image_width), int(lm2.y * image_height)
                
                # Calculate midpoint
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                
                finger_positions[finger_name] = {
                    "midpoint": {"x": mid_x, "y": mid_y},
                    "width": abs(x2 - x1),  # Approximate finger width
                    "angle": np.degrees(np.arctan2(y2 - y1, x2 - x1))  # Finger angle in degrees
                }
            
            return jsonify({
                "message": "Hand detected successfully",
                "image_path": filepath,
                "finger_positions": finger_positions
            })
    
    return jsonify({"error": "Invalid file type"}), 400

# API endpoint for ring try-on
@app.route('/api/try-on', methods=['POST'])
def try_on_ring():
    data = request.form.to_dict()
    
    # Validate input parameters
    if 'hand_image_path' not in data:
        if 'hand_image' not in request.files:
            return jsonify({"error": "No hand image provided"}), 400
        
        # Upload and save hand image
        file = request.files['hand_image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Invalid hand image file"}), 400
        
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        hand_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(hand_image_path)
    else:
        hand_image_path = data['hand_image_path']
        if not os.path.exists(hand_image_path):
            return jsonify({"error": "Hand image file not found"}), 404
    
    # Get ring image path
    if 'ring_id' in data:
        ring_id = data['ring_id']
        if ring_id not in ring_processor.available_rings:
            return jsonify({"error": f"Ring with ID {ring_id} not found"}), 404
        ring_image_path = ring_processor.available_rings[ring_id]
    elif 'ring_image' in request.files:
        # Upload and save ring image
        file = request.files['ring_image']
        if file.filename == '' or not file.filename.lower().endswith('.png'):
            return jsonify({"error": "Ring image must be a PNG file"}), 400
        
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        ring_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(ring_image_path)
    else:
        return jsonify({"error": "No ring specified (provide either ring_id or ring_image)"}), 400
    
    # Process the image
    result_image, error = ring_processor.process_image(hand_image_path, ring_image_path, data)
    
    if error:
        return jsonify({"error": error}), 400
    
    # Save the result image
    result_filename = f"result_{uuid.uuid4()}.jpg"
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    cv2.imwrite(result_path, result_image)
    
    # Return the result
    result_url = f"/api/results/{result_filename}"
    return jsonify({
        "message": "Ring try-on successful",
        "result_image_url": result_url,
        "parameters": data
    })

# API endpoint to get a result image
@app.route('/api/results/<filename>', methods=['GET'])
def get_result(filename):
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename))

# API endpoint to get precise hand measurements
@app.route('/api/measure', methods=['POST'])
def measure_hand():
    if 'hand_image' not in request.files:
        return jsonify({"error": "No hand image provided"}), 400
    
    file = request.files['hand_image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded hand image
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze hand using MediaPipe for detailed measurements
        with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
            image = cv2.imread(filepath)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            if not results.multi_hand_landmarks:
                return jsonify({
                    "error": "No hand detected in the image"
                }), 400
            
            # Extract detailed hand measurements
            image_height, image_width = image.shape[:2]
            landmarks = results.multi_hand_landmarks[0]  # Use first detected hand
            
            # Get all landmark positions
            hand_landmarks = []
            for idx, landmark in enumerate(landmarks.landmark):
                x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                hand_landmarks.append({"x": x, "y": y, "z": landmark.z})
            
            # Calculate finger lengths and widths for ring sizing
            finger_measurements = {}
            
            # Define finger segments for measurements
            finger_segments = {
                "Thumb": [1, 2, 3, 4],
                "Index": [5, 6, 7, 8],
                "Middle": [9, 10, 11, 12],
                "Ring": [13, 14, 15, 16],
                "Pinky": [17, 18, 19, 20]
            }
            
            for finger_name, landmarks_idx in finger_segments.items():
                # Calculate length of each segment
                segments = []
                total_length = 0
                
                for i in range(len(landmarks_idx)-1):
                    idx1 = landmarks_idx[i]
                    idx2 = landmarks_idx[i+1]
                    
                    p1 = np.array([hand_landmarks[idx1]["x"], hand_landmarks[idx1]["y"]])
                    p2 = np.array([hand_landmarks[idx2]["x"], hand_landmarks[idx2]["y"]])
                    
                    length = np.linalg.norm(p2 - p1)
                    segments.append(float(length))
                    total_length += length
                
                # Calculate approximate width at base and middle
                if finger_name == "Ring":
                    # For ring finger, get width at knuckle (MCP joint)
                    mcp_idx = 13  # Ring finger MCP
                    pip_idx = 14  # Ring finger PIP
                    
                    # Width is approximately perpendicular to finger direction
                    dir_vector = np.array([
                        hand_landmarks[pip_idx]["x"] - hand_landmarks[mcp_idx]["x"],
                        hand_landmarks[pip_idx]["y"] - hand_landmarks[mcp_idx]["y"]
                    ])
                    
                    # Normalize and get perpendicular vector
                    dir_vector = dir_vector / np.linalg.norm(dir_vector)
                    perp_vector = np.array([-dir_vector[1], dir_vector[0]])
                    
                    # Approximate width (this is an estimation)
                    width_at_base = float(segments[0] * 0.6)  # Approximate width based on finger segment length
                    
                    finger_measurements[finger_name] = {
                        "segments": segments,
                        "total_length": float(total_length),
                        "width_at_base": width_at_base,
                        "estimated_ring_size": calculate_ring_size(width_at_base)
                    }
                else:
                    finger_measurements[finger_name] = {
                        "segments": segments,
                        "total_length": float(total_length)
                    }
            
            return jsonify({
                "message": "Hand measurements successful",
                "hand_landmarks": hand_landmarks,
                "finger_measurements": finger_measurements
            })
    
    return jsonify({"error": "Invalid file type"}), 400

# Helper function to calculate approximate ring size
def calculate_ring_size(circumference):
    # This is a simplified conversion from circumference to US ring size
    # In a real application, you'd want a more accurate conversion table
    # Circumference is in pixels, so we'd need to convert to mm first
    # For demo purposes, we'll just return a range of sizes
    size_min = max(4, min(13, int(circumference / 20)))
    size_max = size_min + 1
    return f"{size_min}-{size_max} US"

# Cleanup job to remove old files
@app.before_request
def cleanup_old_files():
    # Run this only occasionally (e.g., 1% of requests)
    if np.random.random() < 0.01:
        current_time = time.time()
        # Delete files older than 24 hours
        for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                if os.path.isfile(filepath) and current_time - os.path.getmtime(filepath) > 86400:
                    try:
                        os.remove(filepath)
                    except:
                        pass

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
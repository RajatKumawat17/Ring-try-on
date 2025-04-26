import cv2
import numpy as np
import mediapipe as mp
import os
from math import atan2, degrees

class VirtualRingTryOn:
    def __init__(self, ring_dir="rings"):
        # Initialize MediaPipe Hand module
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load ring images
        self.rings = self.load_rings(ring_dir)
        self.current_ring_idx = 0
        
        # Finger indices for placement (MediaPipe hand landmarks)
        # Index finger: 5 (base), 6, 7, 8 (tip)
        # Middle finger: 9 (base), 10, 11, 12 (tip)
        # Ring finger: 13 (base), 14, 15, 16 (tip)
        # Pinky: 17 (base), 18, 19, 20 (tip)
        self.fingers = {
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        self.target_finger = 'ring'  # Default to ring finger
        
    def load_rings(self, ring_dir):
        """Load ring images from the specified directory"""
        rings = []
        if os.path.exists(ring_dir):
            for filename in os.listdir(ring_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(ring_dir, filename)
                    ring_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if ring_img is not None:
                        rings.append(ring_img)
        
        # Add a default ring if no rings were loaded
        if not rings:
            # Create a simple gold ring as default
            ring_img = np.zeros((100, 100, 4), dtype=np.uint8)
            cv2.ellipse(ring_img, (50, 50), (40, 30), 0, 0, 360, (0, 215, 255, 255), 5)
            rings.append(ring_img)
            
        return rings
    
    def next_ring(self):
        """Switch to the next ring in the collection"""
        self.current_ring_idx = (self.current_ring_idx + 1) % len(self.rings)
    
    def change_target_finger(self, finger_name):
        """Change the target finger for ring placement"""
        if finger_name in self.fingers:
            self.target_finger = finger_name
    
    def overlay_ring(self, frame, landmarks, hand_idx=0):
        """Overlay the current ring on the specified finger"""
        if not landmarks:
            return frame
        
        h, w, _ = frame.shape
        hand_landmarks = landmarks[hand_idx]
        
        # Get coordinates for the target finger
        base_idx = self.fingers[self.target_finger][0]
        mid_idx = self.fingers[self.target_finger][1]
        
        # Get the finger landmarks
        base = (int(hand_landmarks.landmark[base_idx].x * w), 
                int(hand_landmarks.landmark[base_idx].y * h))
        mid = (int(hand_landmarks.landmark[mid_idx].x * w), 
               int(hand_landmarks.landmark[mid_idx].y * h))
        
        # Calculate finger width for ring sizing
        finger_width = int(np.sqrt((base[0] - mid[0])**2 + (base[1] - mid[1])**2) * 1.5)
        
        # Get the angle of the finger
        angle = degrees(atan2(mid[1] - base[1], mid[0] - base[0]))
        
        # Get the current ring and resize it
        ring = self.rings[self.current_ring_idx].copy()
        ring = cv2.resize(ring, (finger_width, finger_width))
        
        # Rotate the ring to match finger orientation
        center = (ring.shape[1] // 2, ring.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        ring = cv2.warpAffine(ring, rotation_matrix, (ring.shape[1], ring.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        
        # Calculate placement position
        placement_x = int(mid[0] - ring.shape[1] // 2)
        placement_y = int(mid[1] - ring.shape[0] // 2)
        
        # Overlay the ring on the frame
        self.overlay_transparent(frame, ring, placement_x, placement_y)
        
        return frame
    
    def overlay_transparent(self, background, overlay, x, y):
        """Overlay a transparent PNG onto the background"""
        if x >= background.shape[1] or y >= background.shape[0]:
            return
            
        h, w = overlay.shape[:2]
        
        # Crop the overlay if it goes outside the background boundaries
        if x < 0:
            overlay = overlay[:, -x:]
            w += x
            x = 0
        if y < 0:
            overlay = overlay[-y:, :]
            h += y
            y = 0
            
        if x + w > background.shape[1]:
            w = background.shape[1] - x
            overlay = overlay[:, :w]
        if y + h > background.shape[0]:
            h = background.shape[0] - y
            overlay = overlay[:h, :]
            
        if overlay.shape[2] < 4:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
            
        # Extract the alpha channel and create masks
        alpha = overlay[:, :, 3] / 255.0
        alpha = np.dstack([alpha, alpha, alpha])
        
        # Extract background and overlay color channels
        bg_region = background[y:y+h, x:x+w, :3]
        overlay_colors = overlay[:, :, :3]
        
        # Composite the overlay onto the background
        background[y:y+h, x:x+w, :3] = bg_region * (1 - alpha) + overlay_colors * alpha
    
    def process_frame(self, frame):
        """Process a video frame and overlay the ring"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Overlay the ring
            frame = self.overlay_ring(frame, results.multi_hand_landmarks)
        
        return frame
    
    def run(self):
        """Run the virtual try-on application"""
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip the frame horizontally for a more natural view
            frame = cv2.flip(frame, 1)
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Display UI instructions
            cv2.putText(processed_frame, "Press 'n' to switch rings", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, "Press 'f' to change finger", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Current finger: {self.target_finger}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, "Press 'q' to quit", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Virtual Ring Try-On', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                self.next_ring()
            elif key == ord('f'):
                # Cycle through fingers
                fingers = list(self.fingers.keys())
                current_idx = fingers.index(self.target_finger)
                next_idx = (current_idx + 1) % len(fingers)
                self.target_finger = fingers[next_idx]
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = VirtualRingTryOn()
    app.run()
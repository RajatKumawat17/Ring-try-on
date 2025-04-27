import os
import sys
import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk


class RingVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ring Visualizer")
        self.root.geometry("1000x700")
        
        # Variables to store paths
        self.hand_image_path = None
        self.ring_image_path = None
        
        # Variables to store images
        self.hand_image = None
        self.ring_image = None
        self.result_image = None
        
        # Initialize finger mapping
        self.finger_map = {
            "Thumb": (mp.solutions.hands.HandLandmark.THUMB_CMC, mp.solutions.hands.HandLandmark.THUMB_MCP),
            "Index": (mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP, mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP),
            "Middle": (mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP),
            "Ring": (mp.solutions.hands.HandLandmark.RING_FINGER_MCP, mp.solutions.hands.HandLandmark.RING_FINGER_PIP),
            "Pinky": (mp.solutions.hands.HandLandmark.PINKY_MCP, mp.solutions.hands.HandLandmark.PINKY_PIP)
        }
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        controls_frame = tk.Frame(main_frame, width=200)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Right panel for image display
        self.display_frame = tk.Frame(main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Hand image selection
        tk.Label(controls_frame, text="Hand Image:").pack(anchor=tk.W, pady=(0, 5))
        self.hand_path_label = tk.Label(controls_frame, text="No file selected", wraplength=180)
        self.hand_path_label.pack(anchor=tk.W, pady=(0, 5))
        hand_btn = tk.Button(controls_frame, text="Select Hand Image", command=self.select_hand_image)
        hand_btn.pack(fill=tk.X, pady=(0, 15))
        
        # Ring image selection
        tk.Label(controls_frame, text="Ring Image:").pack(anchor=tk.W, pady=(0, 5))
        self.ring_path_label = tk.Label(controls_frame, text="No file selected", wraplength=180)
        self.ring_path_label.pack(anchor=tk.W, pady=(0, 5))
        ring_btn = tk.Button(controls_frame, text="Select Ring Image", command=self.select_ring_image)
        ring_btn.pack(fill=tk.X, pady=(0, 15))
        
        # Finger selection dropdown
        tk.Label(controls_frame, text="Select Finger:").pack(anchor=tk.W, pady=(0, 5))
        self.finger_var = tk.StringVar(value="Middle")
        finger_dropdown = ttk.Combobox(controls_frame, textvariable=self.finger_var, 
                                       values=list(self.finger_map.keys()),
                                       state="readonly")
        finger_dropdown.pack(fill=tk.X, pady=(0, 15))
        
        # Ring size adjustment
        tk.Label(controls_frame, text="Ring Width:").pack(anchor=tk.W, pady=(0, 5))
        self.ring_width_var = tk.IntVar(value=100)
        self.ring_width_slider = tk.Scale(controls_frame, from_=20, to=200, orient=tk.HORIZONTAL, 
                                        variable=self.ring_width_var)
        self.ring_width_slider.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(controls_frame, text="Ring Height:").pack(anchor=tk.W, pady=(0, 5))
        self.ring_height_var = tk.IntVar(value=50)
        self.ring_height_slider = tk.Scale(controls_frame, from_=10, to=100, orient=tk.HORIZONTAL, 
                                         variable=self.ring_height_var)
        self.ring_height_slider.pack(fill=tk.X, pady=(0, 15))
        
        # Position adjustment
        tk.Label(controls_frame, text="X Offset:").pack(anchor=tk.W, pady=(0, 5))
        self.x_offset_var = tk.IntVar(value=0)
        self.x_offset_slider = tk.Scale(controls_frame, from_=-100, to=100, orient=tk.HORIZONTAL, 
                                      variable=self.x_offset_var)
        self.x_offset_slider.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(controls_frame, text="Y Offset:").pack(anchor=tk.W, pady=(0, 5))
        self.y_offset_var = tk.IntVar(value=0)
        self.y_offset_slider = tk.Scale(controls_frame, from_=-100, to=100, orient=tk.HORIZONTAL, 
                                      variable=self.y_offset_var)
        self.y_offset_slider.pack(fill=tk.X, pady=(0, 15))
        
        # Process button
        self.process_btn = tk.Button(controls_frame, text="Process Image", command=self.process_image)
        self.process_btn.pack(fill=tk.X, pady=(0, 15))
        self.process_btn.config(state=tk.DISABLED)
        
        # Save button
        self.save_btn = tk.Button(controls_frame, text="Save Result", command=self.save_result)
        self.save_btn.pack(fill=tk.X, pady=(0, 15))
        self.save_btn.config(state=tk.DISABLED)
        
        # Image display
        self.canvas = tk.Canvas(self.display_frame, bg="light gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
    def select_hand_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Hand Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            self.hand_image_path = file_path
            self.hand_path_label.config(text=os.path.basename(file_path))
            self.check_process_ready()
            
            # Preview the hand image
            self.hand_image = cv2.imread(file_path)
            if self.hand_image is not None:
                self.display_image(self.hand_image)
            else:
                messagebox.showerror("Error", "Failed to load the hand image.")
    
    def select_ring_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Ring Image",
            filetypes=[("PNG files", "*.png")]
        )
        
        if file_path:
            self.ring_image_path = file_path
            self.ring_path_label.config(text=os.path.basename(file_path))
            self.check_process_ready()
    
    def check_process_ready(self):
        if self.hand_image_path and self.ring_image_path:
            self.process_btn.config(state=tk.NORMAL)
        else:
            self.process_btn.config(state=tk.DISABLED)
    
    def process_image(self):
        if not self.hand_image_path or not self.ring_image_path:
            messagebox.showerror("Error", "Please select both hand and ring images.")
            return
        
        try:
            # Load the hand image
            hand_image = cv2.imread(self.hand_image_path)
            if hand_image is None:
                messagebox.showerror("Error", "Failed to load the hand image.")
                return
            
            # Load the ring image with alpha channel
            ring_image = cv2.imread(self.ring_image_path, cv2.IMREAD_UNCHANGED)
            if ring_image is None:
                messagebox.showerror("Error", "Failed to load the ring image.")
                return
                
            # Resize the ring image based on slider values
            ring_width = self.ring_width_var.get()
            ring_height = self.ring_height_var.get()
            ring_image = cv2.resize(ring_image, (ring_width, ring_height))
            
            # Get selected finger landmarks
            selected_finger = self.finger_var.get()
            landmark_pair = self.finger_map[selected_finger]
            
            # Get offset values
            x_offset = self.x_offset_var.get()
            y_offset = self.y_offset_var.get()
            
            # Process with MediaPipe
            mp_hands = mp.solutions.hands
            with mp_hands.Hands(static_image_mode=True) as hands:
                hand_image_rgb = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
                results = hands.process(hand_image_rgb)
                
                image_height, image_width = hand_image.shape[:2]
                result_image = hand_image.copy()
                
                if not results.multi_hand_landmarks:
                    messagebox.showerror("Error", "No hand detected in the image.")
                    return
                    
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
            
            # Store the result and display it
            self.result_image = result_image
            self.display_image(result_image)
            self.save_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
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
    
    def get_unique_filename(self, base_path):
        """Generate a unique filename by appending a number if the file already exists."""
        if not os.path.exists(base_path):
            return base_path

        base, ext = os.path.splitext(base_path)
        counter = 1
        while True:
            new_path = f"{base}_{counter}{ext}"
            if not os.path.exists(new_path):
                return new_path
            counter += 1
    
    def save_result(self):
        if self.result_image is None:
            messagebox.showerror("Error", "No processed image to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Result Image",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            unique_path = self.get_unique_filename(file_path)
            success = cv2.imwrite(unique_path, self.result_image)
            if success:
                messagebox.showinfo("Success", f"Image saved to {unique_path}")
            else:
                messagebox.showerror("Error", "Failed to save the image.")
    
    def display_image(self, cv_image):
        # Convert from BGR to RGB
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Calculate new dimensions to fit in the canvas while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1:  # The canvas hasn't been drawn yet
            self.canvas.update()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
        
        img_height, img_width = image_rgb.shape[:2]
        scale = min(canvas_width/img_width, canvas_height/img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        if new_width > 0 and new_height > 0:  # Ensure valid dimensions
            # Resize the image
            resized_image = cv2.resize(image_rgb, (new_width, new_height))
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized_image))
            
            # Clear previous image and display new one
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)


if __name__ == "__main__":
    root = tk.Tk()
    app = RingVisualizerApp(root)
    root.mainloop()
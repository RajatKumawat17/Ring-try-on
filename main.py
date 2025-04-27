import os
import cv2
import mediapipe as mp


def get_unique_filename(base_path):
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

def overlay_image_alpha(background, overlay, x, y, alpha_mask):
    """
    Overlay `overlay` onto `background` at position (x, y) with an alpha mask.

    :param background: Background image.
    :param overlay: Overlay image.
    :param x: X-coordinate for the top-left corner of the overlay on the background.
    :param y: Y-coordinate for the top-left corner of the overlay on the background.
    :param alpha_mask: Alpha channel of the overlay image.
    """
    h, w = overlay.shape[:2]
    
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        raise ValueError("Overlay position is out of bounds.")
    
    roi = background[y:y+h, x:x+w]
    
    for c in range(3):
        roi[:, :, c] = (alpha_mask / 255.0 * overlay[:, :, c] + 
                        (1.0 - alpha_mask / 255.0) * roi[:, :, c])
    
    background[y:y+h, x:x+w] = roi


ring_image_path = 'ring.png'
ring_image = cv2.imread(ring_image_path, cv2.IMREAD_UNCHANGED)

if ring_image is None:
    print("Failed to load the ring image. Check the file path.")
    exit()
ring_image = cv2.resize(ring_image, (100, 50))
hand_image_path = 'hand 2.jpg'
hand_image = cv2.imread(hand_image_path)

if hand_image is None:
    print("Failed to load the hand image. Check the file path.")
    exit()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)
hand_image_rgb = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
results = hands.process(hand_image_rgb)
image_height, image_width = hand_image.shape[:2]
if results.multi_hand_landmarks:
    for landmarks in results.multi_hand_landmarks:

        mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        pip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]


        mcp_x, mcp_y = int(mcp.x * image_width), int(mcp.y * image_height)
        pip_x, pip_y = int(pip.x * image_width), int(pip.y * image_height)


        mid_y = (mcp_y + pip_y) // 2


        print(f"MCP: ({mcp_x}, {mcp_y}), PIP: ({pip_x}, {pip_y}), Midpoint: ({mcp_x}, {mid_y})")


        overlay_image_alpha(
            hand_image,
            ring_image[:, :, :3],
            mcp_x-55,
            mid_y-10,
            ring_image[:, :, 3],
        )
output_path = 'hand_with_ring.jpg'
unique_output_path = get_unique_filename(output_path)
cv2.imwrite(unique_output_path, hand_image)
print(f"Image saved to {unique_output_path}")
hands.close()
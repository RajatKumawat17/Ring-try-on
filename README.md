# Ring-try-on

A simple GUI application for visualizing rings on hand images. This tool uses computer vision to detect hand landmarks and automatically places a ring on the selected finger.

## Features

- Interactive GUI for ring visualization
- Upload hand images and transparent ring images
- Select which finger to place the ring on (Thumb, Index, Middle, Ring, or Pinky)
- Adjust ring size (width and height)
- Fine-tune ring position with X/Y offset controls
- Save the resulting image

## Prerequisites

This application requires the following Python packages:

```
opencv-python
mediapipe
numpy
Pillow
setuptools
```

You can install all dependencies using the included `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone this repository or download the source code
2. Install the required dependencies
3. Run the application

```bash
python main.py
```

## How to Use

1. **Select Hand Image**: Click the "Select Hand Image" button and choose a JPG or PNG image of a hand
2. **Select Ring Image**: Click the "Select Ring Image" button and choose a PNG image with transparency (the ring should have a transparent background)
3. **Choose Finger**: Select which finger to place the ring on from the dropdown menu
4. **Adjust Ring Size**: Use the Width and Height sliders to resize the ring as needed
5. **Position Fine-tuning**: Use the X and Y Offset sliders to fine-tune the position if necessary
6. **Process Image**: Click the "Process Image" button to visualize the ring on the hand
7. **Save Result**: Click the "Save Result" button to save the processed image

## How It Works

The application uses MediaPipe's hand landmark detection to identify key points on the hand. It then places the ring at the appropriate position based on the selected finger. The ring is overlaid onto the hand image using alpha blending to respect transparency.

## Troubleshooting

- **No hand detected**: Ensure the hand is clearly visible in the image. Try using a different hand image with better lighting and contrast.
- **Ring position issues**: Use the X and Y offset sliders to adjust the position. Different hand poses and angles may require different positioning.
- **Ring size issues**: Adjust the Width and Height sliders to better fit the ring to the finger.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- This application uses [MediaPipe](https://mediapipe.dev/) for hand landmark detection
- Built with Python and Tkinter

import os
import cv2
import torch
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ---------------- MODEL LOAD ----------------
print("🚀 Loading AI Models...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Real-ESRGAN Upscaler
model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=2
)
upsampler = RealESRGANer(
    scale=2,
    model_path="RealESRGAN_x2plus.pth",
    model=model,   # ✅ REQUIRED
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)

# GFPGAN Face Enhancer
face_enhancer = GFPGANer(
    model_path="GFPGANv1.3.pth",
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=upsampler
)

print("✅ Models Loaded Successfully!")

# ---------------- HELPERS ----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(filepath):
    img = cv2.imread(filepath)

    if img is None:
        return None

    print("⚙️ Processing Image...")

    # Step 1: Upscale
    output, _ = upsampler.enhance(img, outscale=2)

    # Step 2: Face enhancement
    _, _, final_img = face_enhancer.enhance(
        output,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )

    return final_img

# ---------------- ROUTES ----------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)

            # Process
            output_img = process_image(input_path)

            output_filename = f"enhanced_{filename}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

            cv2.imwrite(output_path, output_img)

            return render_template(
                'index.html',
                original_image=filename,
                enhanced_image=output_filename
            )

    return render_template('index.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(app.config['UPLOAD_FOLDER'], filename),
        as_attachment=True
    )
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)
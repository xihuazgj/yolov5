import subprocess
import uuid
from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS
import detect
import time
app = Flask(__name__, static_folder='runs/detect', static_url_path="/")
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    extension = os.path.splitext(file.filename)[1]
    unique_filename = str(uuid.uuid4()) + extension
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)

    try:
        # 运行对象检测
        t = time.time()
        output_dir = f'exp{t}'
        output_video_path = detect.run(weights='yolov5s.pt',
                   source=file_path,
                   name=output_dir,
                   device='0')
                # 定义输出视频的编码格式
        output_codec = 'h264'

        # 创建输出目录
        # os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], output_dir), exist_ok=True)

        # 转换视频格式
        h264_output_path = os.path.splitext(output_video_path)[0] + '_avc.mp4'
        subprocess.run([
            'ffmpeg',
            '-i', output_video_path,  # 输入文件
            '-c:v', output_codec,  # 视频编解码器
            '-c:a', 'aac',  # 音频编解码器
            h264_output_path  # 输出文件
        ], check=True)
        print('视频格式转换完成!:', h264_output_path)
        # 构建结果URL
        result_filename = os.path.basename(h264_output_path)
        print('结果文件名:', result_filename)
        result_url = f'http://localhost:5000/{output_dir}/{result_filename}'
        return jsonify({'url': result_url})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': 'ffmpeg转码失败', 'details': str(e)}), 500
    except ValueError as e:
        return jsonify({'error': '对象检测失败', 'details': str(e)}), 500
    except Exception as e:
        return jsonify({'error': '内部服务器错误', 'details': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=False)  # 关闭调试模式

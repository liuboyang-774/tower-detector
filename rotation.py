"""
铁塔倾斜检测系统 - 后端 API 服务版 (AKAZE 版)
保留了所有核心姿态计算逻辑，删除了 OpenCV 3D 渲染，新增 Flask 接口与前端 UI 交互。
特征提取已从 SIFT 替换为 AKAZE（C++ 实现，速度更快，无需 GPU）。
"""

import cv2
import math
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# AKAZE 特征提取器（单例，全局复用）
# ============================================================
AKAZE_THRESHOLD = 0.001

def _create_akaze():
    return cv2.AKAZE_create(
        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
        descriptor_size=0,
        descriptor_channels=3,
        threshold=AKAZE_THRESHOLD,
    )

_akaze = _create_akaze()


def extract_akaze_features(img):
    """使用 AKAZE 提取特征点和描述符（替代原 SIFT）"""
    kp, des = _akaze.detectAndCompute(img, None)
    return kp, des

# 初始化 Flask 后端应用
app = Flask(__name__)
CORS(app) # 允许跨域请求，让本地 HTML 可以直接访问

def normalize_azimuth(azimuth):
    return azimuth % 360

def azimuth_to_direction_desc(azimuth):
    azimuth = normalize_azimuth(azimuth)
    if azimuth >= 337.5 or azimuth < 22.5: return "北"
    if azimuth < 67.5:  return "东北"
    if azimuth < 112.5: return "东"
    if azimuth < 157.5: return "东南"
    if azimuth < 202.5: return "南"
    if azimuth < 247.5: return "西南"
    if azimuth < 292.5: return "西"
    return "西北"

def get_local_tilt_direction(delta_pitch_deg, delta_roll_deg, eps=1e-3):
    direction_parts = []
    if abs(delta_pitch_deg) > eps:
        direction_parts.append("前倾" if delta_pitch_deg > 0 else "后倾")
    if abs(delta_roll_deg) > eps:
        direction_parts.append("右倾" if delta_roll_deg > 0 else "左倾")
    return "、".join(direction_parts) if direction_parts else "无明显倾斜"

def calculate_world_tilt(camera_azimuth_deg, pitch_deg, roll_deg):
    if abs(pitch_deg) < 1e-6 and abs(roll_deg) < 1e-6:
        return 0.0, camera_azimuth_deg

    tan_pitch = math.tan(math.radians(pitch_deg))
    tan_roll  = math.tan(math.radians(roll_deg))
    
    total_tilt_rad = math.atan(math.sqrt(tan_pitch**2 + tan_roll**2))
    total_tilt_deg = math.degrees(total_tilt_rad)
    
    local_azimuth_rad = math.atan2(tan_roll, tan_pitch)
    local_azimuth_deg = math.degrees(local_azimuth_rad)
    
    absolute_azimuth = (camera_azimuth_deg + local_azimuth_deg) % 360
    
    return total_tilt_deg, absolute_azimuth

def match_keypoints(img_ref, img_curr, ratio_threshold=0.85):
    kp1, des1 = extract_akaze_features(img_ref)  # ← 替换为 AKAZE
    kp2, des2 = extract_akaze_features(img_curr)  # ← 替换为 AKAZE

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return kp1, kp2, []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2: continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    if len(good_matches) < 8:
        bf_cross = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good_matches = bf_cross.match(des1, des2)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    return kp1, kp2, good_matches

def extract_stable_feature_motion(img_ref, img_curr, ransac_threshold=3.0):
    kp1, kp2, matches = match_keypoints(img_ref, img_curr)
    if len(matches) < 8: return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography, homography_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_threshold)
    if homography is None: return None

    return {
        "kp1": kp1, "kp2": kp2, "matches": matches,
        "pts1": pts1, "pts2": pts2, "homography": homography,
        "static_mask": homography_mask.ravel().astype(bool),
        "inlier_count": int(np.sum(homography_mask)),
        "total_matches": len(matches)
    }

def decompose_homography_to_euler(H, K):
    R_approx = np.linalg.inv(K) @ H @ K
    U, S, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0: R = -R

    euler_angles = cv2.RQDecomp3x3(R)[0]
    return euler_angles[0], euler_angles[1], -euler_angles[2]

def estimate_camera_pose_from_homography(img_ref, img_curr, camera_matrix,
                                         initial_roll_deg=0.0,
                                         initial_pitch_deg=0.0,
                                         initial_yaw_deg=187.0):
    h, w = img_ref.shape[:2]
    img_ref_masked, img_curr_masked = img_ref.copy(), img_curr.copy()
    img_ref_masked[int(h*0.5):, :] = 0
    img_curr_masked[int(h*0.5):, :] = 0

    motion_result = extract_stable_feature_motion(img_ref_masked, img_curr_masked)
    if motion_result is None: return None

    homography = motion_result["homography"]
    delta_pitch, delta_yaw, delta_roll = decompose_homography_to_euler(homography, camera_matrix)

    current_roll_deg = initial_roll_deg + delta_roll
    current_pitch_deg = initial_pitch_deg + delta_pitch
    current_yaw_deg = initial_yaw_deg + delta_yaw

    local_direction = get_local_tilt_direction(delta_pitch, delta_roll)
    confidence = motion_result["inlier_count"] / max(motion_result["total_matches"], 1)

    total_tilt_deg, absolute_azimuth_deg = calculate_world_tilt(initial_yaw_deg, current_pitch_deg, current_roll_deg)

    return {
        "initial_yaw_deg": initial_yaw_deg,                   
        "delta_roll_deg": delta_roll,
        "delta_pitch_deg": delta_pitch,
        "delta_yaw_deg": delta_yaw,
        "current_roll_deg": current_roll_deg,
        "current_pitch_deg": current_pitch_deg,
        "current_yaw_deg": current_yaw_deg,
        "local_direction": local_direction,
        "total_tilt_deg": total_tilt_deg,                     
        "absolute_azimuth_deg": absolute_azimuth_deg,         
        "absolute_direction_desc": azimuth_to_direction_desc(absolute_azimuth_deg), 
        "confidence": confidence,
    }


# ==========================================
# Flask 后端路由：接收前端图片和数据并进行分析
# ==========================================
@app.route('/analyze', methods=['POST'])
def analyze_tilt():
    try:
        # 1. 验证文件是否上传
        if 'baseline' not in request.files or 'tilt' not in request.files:
            return jsonify({'error': '请提供基准图片和倾斜图片'}), 400

        file_ref = request.files['baseline']
        file_curr = request.files['tilt']
        
        # 2. 获取前端传来的初始方位角，默认为 276.0
        initial_yaw_deg = float(request.form.get('initial_yaw_deg', 276.0))

        # 3. 将前端发来的文件流直接转换为 OpenCV 灰度矩阵 (无需保存到本地硬盘)
        np_ref = np.frombuffer(file_ref.read(), np.uint8)
        img_ref = cv2.imdecode(np_ref, cv2.IMREAD_GRAYSCALE)
        
        np_curr = np.frombuffer(file_curr.read(), np.uint8)
        img_curr = cv2.imdecode(np_curr, cv2.IMREAD_GRAYSCALE)

        if img_ref is None or img_curr is None:
            return jsonify({'error': '图片解码失败，请检查文件格式'}), 400

        # 相机内参
        camera_matrix = np.array([
            [1.91947981e+03, 0.00000000e+00, 1.26566418e+03],  
            [0.00000000e+00, 1.91941642e+03, 7.04833607e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ], dtype=np.float64)

        # 4. 执行核心算法
        result = estimate_camera_pose_from_homography(
            img_ref, img_curr, camera_matrix,
            0.0, 0.0, initial_yaw_deg
        )

        if result is None:
            return jsonify({'error': '稳定特征点不足，无法提取姿态数据'}), 400

        # 5. 将计算好的角度返回给前端 UI
        return jsonify({
            'success': True,
            'pitch': result['current_pitch_deg'],
            'roll': result['current_roll_deg'],
            'total_tilt': result['total_tilt_deg'],
            'azimuth': result['absolute_azimuth_deg'],
            'direction_desc': result['absolute_direction_desc']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("==================================================")
    print(" 铁塔姿态分析后台引擎已启动！")
    print(" 请保持此窗口开启，并直接在浏览器中双击打开你的 HTML UI 文件。")
    print("==================================================")
    # 在 5000 端口启动本地后端
    app.run(host='127.0.0.1', port=5000, debug=True)
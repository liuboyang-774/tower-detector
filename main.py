"""
铁塔倾斜检测系统 - 后端 API 服务版 (Docker版)
v2.0 — 全面升级：畸变校正 / 多策略特征匹配 / 塔身状态分类 / 历史震荡分析
"""

import math
import time
import threading
from collections import deque

import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
#  全局配置常量
# ============================================================

# --- 相机标定参数（用户提供的最新标定结果） ---
CAMERA_MATRIX = np.array([
    [1.91449919e+03, 0.00000000e+00, 1.25476234e+03],
    [0.00000000e+00, 1.91837685e+03, 7.16649048e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float64)

DIST_COEFFS = np.array([
    [-5.61757289e-01, 3.56650327e-01, -3.45290566e-04,
     -3.71246347e-04, -1.21195119e-01]
], dtype=np.float64)

# --- 状态分类阈值 ---
TILT_NOISE_THRESHOLD = 0.3       # 低于此值视为测量噪声 (度)
TILT_OSCILLATION_BOUNDARY = 2.0  # 震荡/明显倾斜分界线 (度)
OSCILLATION_STD_THRESHOLD = 0.5  # 标准差超过此值判定为震荡 (度)
MIN_GOOD_MATCHES = 8             # 最少有效匹配点
COLLAPSE_CONSECUTIVE_FRAMES = 3  # 连续多少帧无匹配判定为倒塌
TREND_CONFIRM_FRAMES = 5         # 连续单调帧数确认趋势

# --- 滑动窗口配置 ---
HISTORY_WINDOW_SIZE = 10         # 滑动窗口大小 (帧数)

# --- OSD 水印掩码区域 (像素坐标) ---
# 左上角时间戳区域
OSD_TOP_LEFT = (0, 0, 620, 90)
# 右下角 Camera 标识区域 (相对于右下角的偏移，运行时根据图像尺寸计算)
OSD_BOTTOM_RIGHT_OFFSET = (300, 70)


# ============================================================
#  畸变校正模块 — 预计算映射表，每帧只做 remap
# ============================================================
class UndistortEngine:
    """镜头畸变校正引擎，启动时预计算映射表"""

    def __init__(self, camera_matrix, dist_coeffs, alpha=0):
        self._K = camera_matrix
        self._D = dist_coeffs
        self._alpha = alpha
        self._map1 = None
        self._map2 = None
        self._new_K = None
        self._roi = None
        self._initialized_shape = None

    def _ensure_init(self, h, w):
        """根据图像尺寸初始化/重新初始化映射表"""
        shape_key = (h, w)
        if self._initialized_shape == shape_key:
            return

        self._new_K, self._roi = cv2.getOptimalNewCameraMatrix(
            self._K, self._D, (w, h), self._alpha, (w, h)
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self._K, self._D, None, self._new_K, (w, h), cv2.CV_16SC2
        )
        self._initialized_shape = shape_key
        print(f"[UndistortEngine] 映射表已初始化: {w}x{h}, "
              f"新焦距 fx={self._new_K[0,0]:.1f} fy={self._new_K[1,1]:.1f}")

    def undistort(self, img):
        """对图像执行去畸变"""
        h, w = img.shape[:2]
        self._ensure_init(h, w)
        return cv2.remap(img, self._map1, self._map2, cv2.INTER_LINEAR)

    @property
    def new_camera_matrix(self):
        """去畸变后的新内参矩阵"""
        return self._new_K if self._new_K is not None else self._K


# ============================================================
#  历史数据管理器 — 滑动窗口 + 线程安全
# ============================================================
class TiltHistoryManager:
    """多塔倾斜历史数据管理，支持滑动窗口统计分析"""

    def __init__(self, window_size=HISTORY_WINDOW_SIZE):
        self._window_size = window_size
        self._towers = {}      # tower_id -> deque of records
        self._lock = threading.Lock()

    def _get_tower(self, tower_id):
        if tower_id not in self._towers:
            self._towers[tower_id] = deque(maxlen=self._window_size)
        return self._towers[tower_id]

    def add_record(self, tower_id, record):
        """
        添加一条测量记录
        record: dict with keys: timestamp, total_tilt, pitch, roll,
                azimuth, match_count, success(bool)
        """
        with self._lock:
            buf = self._get_tower(tower_id)
            record['timestamp'] = time.time()
            buf.append(record)

    def get_stats(self, tower_id):
        """获取滑动窗口统计"""
        with self._lock:
            buf = self._get_tower(tower_id)
            if not buf:
                return None

            successful = [r for r in buf if r.get('success', False)]
            failed = [r for r in buf if not r.get('success', False)]

            if not successful:
                return {
                    'window_size': len(buf),
                    'successful_frames': 0,
                    'failed_frames': len(failed),
                    'consecutive_failures': len(buf),
                }

            tilts = [r['total_tilt'] for r in successful]
            pitches = [r['pitch'] for r in successful]
            rolls = [r['roll'] for r in successful]
            
            # 关键修复：使用真实空间方向计算标准差，而非标量幅度。
            # 防止铁塔出现前后对称摆动时，由于总幅度(total_tilt)不变导致算出的标准差为 0 的误判。
            std_tilt = float(max(np.std(pitches), np.std(rolls)))

            # 计算趋势：连续单调递增/递减的帧数
            trend = self._analyze_trend(tilts, std_tilt)

            return {
                'window_size': len(buf),
                'successful_frames': len(successful),
                'failed_frames': len(failed),
                'mean_tilt': float(np.mean(tilts)),
                'std_tilt': std_tilt,
                'max_tilt': float(np.max(tilts)),
                'min_tilt': float(np.min(tilts)),
                'latest_tilt': tilts[-1],
                'trend': trend,
                'consecutive_failures': self._count_consecutive_failures(buf),
            }

    def _analyze_trend(self, tilts, std_tilt):
        """分析倾斜趋势"""
        if len(tilts) < 3:
            return 'insufficient_data'

        # 计算连续递增/递减
        increasing = 0
        decreasing = 0
        for i in range(1, len(tilts)):
            if tilts[i] > tilts[i-1] + 0.05:  # 加小容差避免噪声
                increasing += 1
                decreasing = 0
            elif tilts[i] < tilts[i-1] - 0.05:
                decreasing += 1
                increasing = 0
            # 如果差值在容差内，不重置计数

        if increasing >= TREND_CONFIRM_FRAMES:
            return 'increasing'
        elif decreasing >= TREND_CONFIRM_FRAMES:
            return 'decreasing'

        # 检查是否在波动
        if std_tilt > OSCILLATION_STD_THRESHOLD:
            return 'oscillating'

        return 'stable'

    def _count_consecutive_failures(self, buf):
        """从最新记录向前数连续失败帧数"""
        count = 0
        for r in reversed(buf):
            if not r.get('success', False):
                count += 1
            else:
                break
        return count

    def reset(self, tower_id):
        """清空指定塔的历史数据"""
        with self._lock:
            if tower_id in self._towers:
                self._towers[tower_id].clear()

    def get_all_tower_ids(self):
        with self._lock:
            return list(self._towers.keys())


# ============================================================
#  塔身状态分类器
# ============================================================
class TowerStateClassifier:
    """基于当前测量 + 历史数据的三级状态分类"""

    # 状态码定义
    NORMAL = 'NORMAL'
    MINOR_OFFSET = 'MINOR_OFFSET'
    OSCILLATING_SMALL = 'OSCILLATING_SMALL'
    OSCILLATING_LARGE = 'OSCILLATING_LARGE'
    TILTING = 'TILTING'
    TEMPORARY_OCCLUSION = 'TEMPORARY_OCCLUSION'
    COLLAPSED = 'COLLAPSED'

    STATUS_INFO = {
        NORMAL:              {'label': '正常',        'severity': 'ok'},
        MINOR_OFFSET:        {'label': '微小偏移',    'severity': 'ok'},
        OSCILLATING_SMALL:   {'label': '小范围震荡',  'severity': 'warning'},
        OSCILLATING_LARGE:   {'label': '大范围震荡',  'severity': 'warning'},
        TILTING:             {'label': '明显倾斜',    'severity': 'critical'},
        TEMPORARY_OCCLUSION: {'label': '临时遮挡',    'severity': 'info'},
        COLLAPSED:           {'label': '塔身倒塌/遮挡', 'severity': 'critical'},
    }

    @classmethod
    def classify(cls, current_result, history_stats):
        """
        分类塔身状态

        Args:
            current_result: 当前帧计算结果 (None 表示匹配失败)
            history_stats:  滑动窗口统计数据

        Returns:
            dict: {code, label, severity, detail}
        """
        # --- 路径1：当前帧匹配失败 ---
        if current_result is None:
            consecutive_failures = (history_stats or {}).get(
                'consecutive_failures', 1
            )
            if consecutive_failures >= COLLAPSE_CONSECUTIVE_FRAMES:
                return cls._make_status(
                    cls.COLLAPSED,
                    f"连续 {consecutive_failures} 帧无法检测到有效特征匹配，"
                    f"塔身可能已倒塌或相机严重遮挡"
                )
            else:
                return cls._make_status(
                    cls.TEMPORARY_OCCLUSION,
                    f"当前帧匹配失败（连续 {consecutive_failures} 帧），"
                    f"可能为临时遮挡或光照剧变"
                )

        # --- 路径2：当前帧匹配成功，根据倾斜角分类 ---
        tilt = current_result['total_tilt']

        # 2a. 倾斜角 < 噪声阈值 → 正常
        if tilt < TILT_NOISE_THRESHOLD:
            return cls._make_status(
                cls.NORMAL,
                f"倾斜角 {tilt:.3f}° 在测量噪声范围内"
            )

        # 2b. 噪声阈值 ≤ 倾斜角 < 震荡分界
        if tilt < TILT_OSCILLATION_BOUNDARY:
            if history_stats and history_stats.get('successful_frames', 0) >= 3:
                std = history_stats.get('std_tilt', 0)
                if std > OSCILLATION_STD_THRESHOLD:
                    return cls._make_status(
                        cls.OSCILLATING_SMALL,
                        f"倾斜角 {tilt:.2f}°，滑动窗口标准差 {std:.3f}° "
                        f"超过阈值 {OSCILLATION_STD_THRESHOLD}°，判定为小范围震荡"
                    )
            return cls._make_status(
                cls.MINOR_OFFSET,
                f"倾斜角 {tilt:.2f}°，在允许范围内"
            )

        # 2c. 倾斜角 ≥ 震荡分界 → 大角度区域
        if history_stats and history_stats.get('successful_frames', 0) >= 3:
            trend = history_stats.get('trend', 'stable')
            std = history_stats.get('std_tilt', 0)
            mean = history_stats.get('mean_tilt', tilt)
            min_t = history_stats.get('min_tilt', tilt)
            max_t = history_stats.get('max_tilt', tilt)

            if trend == 'increasing':
                return cls._make_status(
                    cls.TILTING,
                    f"倾斜角 {tilt:.2f}°，且趋势持续增大"
                    f"（窗口均值 {mean:.2f}°, 最大 {max_t:.2f}°），"
                    f"塔身可能正在发生结构性倾斜"
                )

            if trend == 'oscillating' or std > OSCILLATION_STD_THRESHOLD:
                return cls._make_status(
                    cls.OSCILLATING_LARGE,
                    f"倾斜角 {tilt:.2f}°，窗口内 "
                    f"{min_t:.2f}°~{max_t:.2f}° 波动，"
                    f"标准差 {std:.2f}°，判定为大范围震荡"
                )

        # 默认：角度大但历史数据不足，保守判定为倾斜
        return cls._make_status(
            cls.TILTING,
            f"倾斜角 {tilt:.2f}°，超过安全阈值 "
            f"{TILT_OSCILLATION_BOUNDARY}°"
        )

    @classmethod
    def _make_status(cls, code, detail):
        info = cls.STATUS_INFO[code]
        return {
            'code': code,
            'label': info['label'],
            'severity': info['severity'],
            'detail': detail,
        }


# ============================================================
#  OSD 水印掩码
# ============================================================
def create_osd_mask(h, w):
    """创建 OSD 水印掩码，水印区域为 0，其余为 255"""
    mask = np.ones((h, w), dtype=np.uint8) * 255

    # 左上角时间戳
    x1, y1, x2, y2 = OSD_TOP_LEFT
    x2 = min(x2, w)
    y2 = min(y2, h)
    mask[y1:y2, x1:x2] = 0

    # 右下角 Camera 标识
    bw, bh = OSD_BOTTOM_RIGHT_OFFSET
    mask[max(0, h - bh):h, max(0, w - bw):w] = 0

    return mask


# ============================================================
#  方位角转方向
# ============================================================
def azimuth_to_direction(azimuth):
    azimuth = azimuth % 360.0
    if azimuth < 0:
        azimuth += 360.0
    if azimuth >= 337.5 or azimuth < 22.5:
        return "北"
    if azimuth < 67.5:
        return "东北"
    if azimuth < 112.5:
        return "东"
    if azimuth < 157.5:
        return "东南"
    if azimuth < 202.5:
        return "南"
    if azimuth < 247.5:
        return "西南"
    if azimuth < 292.5:
        return "西"
    return "西北"


# ============================================================
#  世界坐标系倾斜计算
# ============================================================
def calculate_world_tilt(camera_azimuth_deg, pitch_deg, roll_deg):
    if abs(pitch_deg) < 1e-6 and abs(roll_deg) < 1e-6:
        return 0.0, camera_azimuth_deg

    tan_pitch = math.tan(pitch_deg * math.pi / 180.0)
    tan_roll = math.tan(roll_deg * math.pi / 180.0)

    total_tilt_deg = math.atan(
        math.sqrt(tan_pitch**2 + tan_roll**2)
    ) * 180.0 / math.pi

    local_azimuth_deg = math.atan2(tan_roll, tan_pitch) * 180.0 / math.pi
    absolute_azimuth_deg = (camera_azimuth_deg + local_azimuth_deg) % 360.0
    if absolute_azimuth_deg < 0:
        absolute_azimuth_deg += 360.0

    return total_tilt_deg, absolute_azimuth_deg


# ============================================================
#  核心特征匹配引擎 — 多策略级联
# ============================================================
def extract_and_match(img_ref, img_curr, mask_ref, mask_curr):
    """
    多策略特征匹配：
      1. AKAZE（调优参数）
      2. 如果 AKAZE 匹配不足，fallback 到 SIFT
      3. 双向交叉验证 + Lowe's Ratio Test

    Returns:
        (pts1, pts2, match_count, method_used) 或 (None, None, 0, None) 匹配失败
    """
    methods = [
        ('AKAZE', _match_akaze),
        ('SIFT', _match_sift),
    ]

    for method_name, match_fn in methods:
        pts1, pts2, match_count = match_fn(
            img_ref, img_curr, mask_ref, mask_curr
        )
        if pts1 is not None and match_count >= MIN_GOOD_MATCHES:
            print(f"[FeatureMatch] {method_name} 成功: "
                  f"{match_count} 个有效匹配点")
            return pts1, pts2, match_count, method_name

        print(f"[FeatureMatch] {method_name} 匹配点不足: "
              f"{match_count}，尝试下一策略")

    return None, None, 0, None


def _match_akaze(img_ref, img_curr, mask_ref, mask_curr):
    """AKAZE 特征匹配（调优参数）"""
    akaze = cv2.AKAZE_create(
        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
        threshold=0.0005,         # 降低阈值，检测更多特征
        nOctaves=4,
        nOctaveLayers=5,          # 增加层数
    )

    kp1, des1 = akaze.detectAndCompute(img_ref, mask_ref)
    kp2, des2 = akaze.detectAndCompute(img_curr, mask_curr)

    print(f"[AKAZE] 特征点: ref={len(kp1) if kp1 else 0}, "
          f"curr={len(kp2) if kp2 else 0}")

    if des1 is None or des2 is None or len(kp1) < MIN_GOOD_MATCHES or len(kp2) < MIN_GOOD_MATCHES:
        return None, None, 0

    return _cross_check_match(kp1, des1, kp2, des2,
                              norm_type=cv2.NORM_HAMMING,
                              ratio_threshold=0.70)


def _match_sift(img_ref, img_curr, mask_ref, mask_curr):
    """SIFT 特征匹配（作为 fallback）"""
    sift = cv2.SIFT_create(
        nfeatures=2000,
        contrastThreshold=0.03,
        edgeThreshold=15,
    )

    kp1, des1 = sift.detectAndCompute(img_ref, mask_ref)
    kp2, des2 = sift.detectAndCompute(img_curr, mask_curr)

    print(f"[SIFT] 特征点: ref={len(kp1) if kp1 else 0}, "
          f"curr={len(kp2) if kp2 else 0}")

    if des1 is None or des2 is None or len(kp1) < MIN_GOOD_MATCHES or len(kp2) < MIN_GOOD_MATCHES:
        return None, None, 0

    return _cross_check_match(kp1, des1, kp2, des2,
                              norm_type=cv2.NORM_L2,
                              ratio_threshold=0.70)


def _cross_check_match(kp1, des1, kp2, des2, norm_type, ratio_threshold):
    """
    双向匹配 + Lowe's Ratio Test + 交叉验证

    Returns:
        (pts1, pts2, match_count) 或 (None, None, 0)
    """
    matcher = cv2.BFMatcher(norm_type)

    # 正向匹配: des1 → des2
    knn_fwd = matcher.knnMatch(des1, des2, k=2)
    good_fwd = {}
    for match in knn_fwd:
        if len(match) == 2 and match[0].distance < ratio_threshold * match[1].distance:
            good_fwd[match[0].queryIdx] = match[0].trainIdx

    # 反向匹配: des2 → des1
    knn_bwd = matcher.knnMatch(des2, des1, k=2)
    good_bwd = {}
    for match in knn_bwd:
        if len(match) == 2 and match[0].distance < ratio_threshold * match[1].distance:
            good_bwd[match[0].queryIdx] = match[0].trainIdx

    # 交叉验证：正向匹配 A→B 和反向匹配 B→A 一致的才保留
    cross_checked = []
    for q_idx, t_idx in good_fwd.items():
        if t_idx in good_bwd and good_bwd[t_idx] == q_idx:
            cross_checked.append((q_idx, t_idx))

    if len(cross_checked) < MIN_GOOD_MATCHES:
        return None, None, len(cross_checked)

    pts1 = np.float32([kp1[q].pt for q, t in cross_checked]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[t].pt for q, t in cross_checked]).reshape(-1, 1, 2)

    return pts1, pts2, len(cross_checked)


# ============================================================
#  核心算法：倾斜角计算
# ============================================================
def run_algo(img_ref, img_curr, initial_yaw, undistort_engine):
    """
    核心倾斜检测算法

    流程：去畸变 → OSD掩码 → 特征匹配 → 单应矩阵 → 欧拉角分解

    Returns:
        (result_dict, error_msg)
    """
    # 1. 去畸变
    img_ref_ud = undistort_engine.undistort(img_ref)
    img_curr_ud = undistort_engine.undistort(img_curr)

    # 2. 创建 OSD 掩码
    h, w = img_ref_ud.shape[:2]
    mask_ref = create_osd_mask(h, w)
    h2, w2 = img_curr_ud.shape[:2]
    mask_curr = create_osd_mask(h2, w2)

    # 3. 多策略特征匹配
    pts1, pts2, match_count, method = extract_and_match(
        img_ref_ud, img_curr_ud, mask_ref, mask_curr
    )

    if pts1 is None:
        return None, (f"[安全拦截] 所有特征匹配策略均失败，"
                      f"有效匹配点数: {match_count}")

    # 4. RANSAC 求解单应矩阵
    H, inlier_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0,
                                         maxIters=5000, confidence=0.995)
    if H is None or H.shape != (3, 3):
        return None, "[安全拦截] 单应矩阵拟合失败（RANSAC 未收敛）"

    # 统计内点数
    inlier_count = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
    print(f"[Homography] RANSAC 内点: {inlier_count}/{match_count}")

    # 5. 从单应矩阵分解旋转
    K = undistort_engine.new_camera_matrix
    K_inv = np.linalg.inv(K)
    R_approx = K_inv @ H @ K

    # SVD 正交化
    U, S, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R

    # 6. 欧拉角分解
    delta_pitch = math.atan2(-R[2, 1], R[2, 2]) * 180.0 / math.pi
    delta_roll = math.atan2(-R[1, 0], R[0, 0]) * 180.0 / math.pi

    # 7. 计算世界坐标系倾斜
    total_tilt, absolute_azimuth = calculate_world_tilt(
        initial_yaw, delta_pitch, delta_roll
    )

    return {
        'success': True,
        'pitch': round(delta_pitch, 4),
        'roll': round(delta_roll, 4),
        'total_tilt': round(total_tilt, 4),
        'azimuth': round(absolute_azimuth, 4),
        'direction_desc': azimuth_to_direction(absolute_azimuth),
        'match_count': match_count,
        'inlier_count': inlier_count,
        'match_method': method,
    }, None


# ============================================================
#  Flask 应用初始化
# ============================================================
app = Flask(__name__)
CORS(app)

# 全局单例
undistort_engine = UndistortEngine(CAMERA_MATRIX, DIST_COEFFS, alpha=0)
history_manager = TiltHistoryManager(window_size=HISTORY_WINDOW_SIZE)


# ============================================================
#  API 路由
# ============================================================

@app.route('/analyze', methods=['POST'])
def analyze_tilt():
    """
    主分析接口

    参数:
        - baseline: 基准图片文件
        - tilt: 当前倾斜图片文件
        - initial_yaw_deg: 初始方位角 (默认 276.0)
        - tower_id: 塔编号 (默认 "default")
    """
    try:
        # 1. 验证文件上传
        if 'baseline' not in request.files or 'tilt' not in request.files:
            return jsonify({
                'error': '请提供基准图片(baseline)和倾斜图片(tilt)'
            }), 400

        file_ref = request.files['baseline']
        file_curr = request.files['tilt']

        # 2. 获取参数
        initial_yaw_deg = float(request.form.get('initial_yaw_deg', 276.0))
        tower_id = request.form.get('tower_id', 'default')

        # 3. 解码图片
        np_ref = np.frombuffer(file_ref.read(), np.uint8)
        img_ref = cv2.imdecode(np_ref, cv2.IMREAD_GRAYSCALE)

        np_curr = np.frombuffer(file_curr.read(), np.uint8)
        img_curr = cv2.imdecode(np_curr, cv2.IMREAD_GRAYSCALE)

        if img_ref is None or img_curr is None:
            return jsonify({
                'error': '[致命错误] 图片解码失败，请检查文件格式'
            }), 400

        # 4. 执行核心算法
        result, error_msg = run_algo(
            img_ref, img_curr, initial_yaw_deg, undistort_engine
        )

        # 5. 记录到历史
        if result is not None:
            history_manager.add_record(tower_id, {
                'success': True,
                'total_tilt': result['total_tilt'],
                'pitch': result['pitch'],
                'roll': result['roll'],
                'azimuth': result['azimuth'],
                'match_count': result['match_count'],
            })
        else:
            history_manager.add_record(tower_id, {
                'success': False,
                'error': error_msg,
            })

        # 6. 获取历史统计
        stats = history_manager.get_stats(tower_id)

        # 7. 状态分类
        status = TowerStateClassifier.classify(result, stats)

        # 8. 构建响应
        if result is None:
            return jsonify({
                'error': error_msg,
                'status': status,
                'history_stats': stats,
            }), 400

        result['status'] = status
        result['history_stats'] = stats

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def get_tower_status():
    """
    查询指定塔的当前状态和历史统计

    参数:
        - tower_id: 塔编号 (默认 "default")
    """
    tower_id = request.args.get('tower_id', 'default')
    stats = history_manager.get_stats(tower_id)

    if stats is None:
        return jsonify({
            'tower_id': tower_id,
            'message': '暂无该塔的历史数据',
        })

    # 用最近一次结果做状态分类
    latest_result = None
    if stats.get('successful_frames', 0) > 0:
        latest_result = {'total_tilt': stats['latest_tilt']}

    status = TowerStateClassifier.classify(latest_result, stats)

    return jsonify({
        'tower_id': tower_id,
        'status': status,
        'history_stats': stats,
    })


@app.route('/reset', methods=['POST'])
def reset_tower():
    """
    清空指定塔的历史数据

    参数:
        - tower_id: 塔编号 (默认 "default")
    """
    tower_id = request.form.get('tower_id',
                                request.args.get('tower_id', 'default'))
    history_manager.reset(tower_id)
    return jsonify({
        'tower_id': tower_id,
        'message': f'塔 {tower_id} 的历史数据已清空',
    })


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'version': '2.0',
        'message': '铁塔倾斜检测服务运行中 (v2.0: 畸变校正 + 多策略匹配 + 状态分类)',
        'features': [
            '镜头畸变校正',
            'AKAZE+SIFT 双策略特征匹配',
            '双向交叉验证',
            'OSD 水印掩码',
            '塔身状态分类 (正常/震荡/倾斜/倒塌)',
            '滑动窗口历史分析',
        ],
        'active_towers': history_manager.get_all_tower_ids(),
    })


# ============================================================
#  启动
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  铁塔倾斜检测系统 v2.0")
    print("  畸变校正 | 多策略匹配 | 状态分类 | 历史分析")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)

import cv2
import os
import sys


def extract_frames(video_path, output_folder="picture_cut", interval_sec=5):
    """
    从视频中每隔指定秒数截取一帧并保存

    Args:
        video_path: 视频文件路径
        output_folder: 输出文件夹名称
        interval_sec: 截帧间隔（秒）
    """
    # 1. 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ 已创建输出文件夹: {output_folder}")

    # 2. 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return

    # 3. 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"📹 视频信息:")
    print(f"   FPS: {fps:.2f}")
    print(f"   总帧数: {total_frames}")
    print(f"   时长: {duration:.2f} 秒 ({duration/60:.1f} 分钟)")
    print(f"   截帧间隔: {interval_sec} 秒")
    print(f"   预计截取: {int(duration // interval_sec) + 1} 帧")
    print("-" * 40)

    # 4. 计算需要截取的帧位置
    frame_interval = int(fps * interval_sec)
    saved_count = 0
    current_frame = 0

    while True:
        # 跳转到目标帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            break

        # 5. 保存帧图片
        timestamp = current_frame / fps
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        filename = f"frame_{saved_count:04d}_{minutes:02d}m{seconds:02d}s.jpg"
        save_path = os.path.join(output_folder, filename)

        cv2.imwrite(save_path, frame)
        saved_count += 1
        print(f"   ✅ [{saved_count}] 已保存: {filename} (时间点: {minutes:02d}:{seconds:02d})")

        # 移动到下一个截取点
        current_frame += frame_interval

    # 6. 释放资源
    cap.release()
    print("-" * 40)
    print(f"🎉 完成! 共截取 {saved_count} 帧，保存在 '{output_folder}' 文件夹中")


if __name__ == "__main__":
    # ========== 修改这里 ==========
    VIDEO_PATH = r"C:\Users\admin\Desktop\tietapingheng\718.mp4"      # ← 替换为你的视频文件名
    OUTPUT_FOLDER = "picture_cut"          # 输出文件夹名
    INTERVAL = 5                    # 截帧间隔（秒）
    # ==============================

    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]

    extract_frames(VIDEO_PATH, OUTPUT_FOLDER, INTERVAL)
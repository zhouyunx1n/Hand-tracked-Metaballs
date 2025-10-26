import math, random, time
import numpy as np
import pygame as pg

# --- OpenCV & MediaPipe ---
import cv2
import mediapipe as mp

# ---------------- 基本设置 ----------------
W, H = 1000, 700
FPS = 60

pg.init()
screen = pg.display.set_mode((W, H))
pg.display.set_caption("Hand-tracked Metaballs")
clock = pg.time.Clock()
font = pg.font.SysFont("Segoe UI", 16)
big_font = pg.font.SysFont("Segoe UI", 26, bold=True)

# 摄像头小窗（右上角）
CAM_W, CAM_H = 360, 270
CAM_RECT = pg.Rect(W - CAM_W - 16, 16, CAM_W, CAM_H)

# ---------------- 摄像头 ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------- MediaPipe Hands ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# 画一个白色小方块
def draw_square(surf, center, size=10, color=(255,255,255)):
    x, y = int(center[0]-size/2), int(center[1]-size/2)
    pg.draw.rect(surf, color, pg.Rect(x, y, size, size), border_radius=2)

# ---------------- 软圆渲染（发光气泡） ----------------
def draw_soft_circle(surf, center, radius, color_inner, color_outer):
    x, y = center
    layers = max(4, radius // 6)
    for i in range(layers, 0, -1):
        t = i / layers
        r = int(radius * (0.35 + 0.65 * t))
        a = int(255 * (t ** 2))
        col = (
            int(color_outer[0] * (1 - t) + color_inner[0] * t),
            int(color_outer[1] * (1 - t) + color_inner[1] * t),
            int(color_outer[2] * (1 - t) + color_inner[2] * t),
            a
        )
        pg.draw.circle(surf, col, (int(x), int(y)), max(1, r))

def lerp(a, b, t): return a + (b - a) * t

PALETTE = [
    ((255, 230, 140), (60, 255, 140)),
    ((255, 160, 200), (120, 255, 200)),
    ((255, 210, 120), (90, 255, 180)),
    ((180, 220, 255), (120, 240, 255))
]

class Blob:
    def __init__(self, x, y, r):
        self.x, self.y, self.r = x, y, r
        self.vx = random.uniform(-40, 40)
        self.vy = random.uniform(-40, 40)
        self.inner, self.outer = random.choice(PALETTE)
        self.pulse = random.uniform(0, 1000)
        self.base_r = r

    def update(self, dt, attract=None, energy=0.0):
        self.vx += math.sin(self.pulse + time.time()*0.9) * 2.0 * dt
        self.vy += math.cos(self.pulse + time.time()*0.7) * 2.0 * dt

        if attract is not None:
            ax, ay = attract
            dx, dy = ax - self.x, ay - self.y
            dist = math.hypot(dx, dy) + 1e-5
            force = 60 + 320 * energy
            self.vx += (dx / dist) * force * dt
            self.vy += (dy / dist) * force * dt
            target_r = self.base_r * (1.0 + 0.65 * energy)
            self.r = lerp(self.r, target_r, 4.0 * dt)
        else:
            self.r = lerp(self.r, self.base_r, 1.5 * dt)

        self.vx *= (1 - 0.9 * dt)
        self.vy *= (1 - 0.9 * dt)
        self.x += self.vx * dt
        self.y += self.vy * dt

        margin = 40
        if self.x < margin: self.vx += (margin - self.x) * 4 * dt
        if self.x > W - margin: self.vx -= (self.x - (W - margin)) * 4 * dt
        if self.y < margin: self.vy += (margin - self.y) * 4 * dt
        if self.y > H - margin: self.vy -= (self.y - (H - margin)) * 4 * dt

    def draw(self, surf):
        draw_soft_circle(surf, (self.x, self.y), int(self.r), self.inner, self.outer)

# 初始化气泡
blobs = []
for _ in range(9):
    b = Blob(
        x=random.uniform(W*0.15, W*0.55),
        y=random.uniform(H*0.25, H*0.75),
        r=random.randint(28, 60)
    )
    blobs.append(b)
controller = blobs[0]

# 指尖轨迹（绘在相机预览上）
trail = []          # 存储最近 N 帧的指尖点（相机小窗坐标）
TRAIL_MAX = 50
last_tip_world = None   # 上一帧在大画布上的指尖点，用于速度计算

def normalize_landmark(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def draw_smooth_polyline(surf, pts, color=(255,255,255), width=2):
    if len(pts) < 2: return
    pg.draw.lines(surf, color, False, pts, width)

running = True
while running:
    dt = clock.tick(FPS) / 1000.0

    for e in pg.event.get():
        if e.type == pg.QUIT:
            running = False
        if e.type == pg.KEYDOWN:
            if e.key == pg.K_s:
                fname = f"screenshot_{int(time.time())}.png"
                pg.image.save(screen, fname)
                print("Saved:", fname)
            if e.key == pg.K_q:
                running = False

    # 读相机帧
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h_cam, w_cam = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe 识别
    res = hands.process(rgb)

    # 生成相机小窗 Surface
    small = cv2.resize(rgb, (CAM_W, CAM_H))
    frame_surf = pg.image.frombuffer(small.tobytes(), (CAM_W, CAM_H), "RGB")

    # ------- 解析手部关键点 -------
    attract_world = None  # 映射到大画布上的吸引点
    energy = 0.0          # 用于放大/扰动

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        # 关键点：食指指尖(8)、拇指指尖(4)、中指指尖(12)、无名指尖(16)、小指指尖(20)、手腕(0)
        idx = mp_hands.HandLandmark
        ids = [idx.INDEX_FINGER_TIP, idx.THUMB_TIP, idx.MIDDLE_FINGER_TIP, idx.RING_FINGER_TIP, idx.PINKY_TIP, idx.WRIST]

        # 相机原始坐标（用于预览叠加）
        pts_cam = []
        for i in ids:
            x = int(hand.landmark[i].x * CAM_W)
            y = int(hand.landmark[i].y * CAM_H)
            pts_cam.append((x, y))

        # 预览：画小白方块
        for p in pts_cam:
            draw_square(frame_surf, p, size=10, color=(255,255,255))

        # 预览：绘制闭合曲线（用指尖们连线）
        curve_pts = pts_cam[:-1]  # 不含腕
        if len(curve_pts) >= 3:
            draw_smooth_polyline(frame_surf, curve_pts + [curve_pts[0]], color=(255,255,255), width=2)

        # 预览：黄色圆点（食指指尖）
        tip_cam = pts_cam[0]
        pg.draw.circle(frame_surf, (255, 210, 0), tip_cam, 8)

        # 记录轨迹
        trail.append(tip_cam)
        if len(trail) > TRAIL_MAX: trail.pop(0)
        if len(trail) >= 2:
            draw_smooth_polyline(frame_surf, trail, color=(255,255,255), width=1)

        # --- 将食指指尖映射到大画布（靠左 75% 区域） ---
        tip_norm_x = hand.landmark[idx.INDEX_FINGER_TIP].x
        tip_norm_y = hand.landmark[idx.INDEX_FINGER_TIP].y
        ax = tip_norm_x * (W * 0.75)
        ay = tip_norm_y * H
        attract_world = (ax, ay)

        # --- 估算能量：指尖速度 + 手指开合 ---
        # 指尖速度
        speed = 0.0
        if last_tip_world is not None:
            dx = ax - last_tip_world[0]
            dy = ay - last_tip_world[1]
            speed = min(1.0, math.hypot(dx, dy) / 25.0)  # 调参

        # 手指开合：拇指指尖与食指指尖距离（相机坐标下）
        thumb_tip = hand.landmark[idx.THUMB_TIP]
        d_open = math.hypot(
            (thumb_tip.x - tip_norm_x) * w_cam,
            (thumb_tip.y - tip_norm_y) * h_cam
        )
        d_norm = np.clip(d_open / 180.0, 0.0, 1.0)

        energy = np.clip(0.6*speed + 0.6*d_norm, 0.0, 1.0)
        last_tip_world = (ax, ay)

    # ----------------- 绘制主画面 -----------------
    screen.fill((244, 246, 248))

    # 更新气泡
    for b in blobs:
        b.update(dt, attract=attract_world if b is controller else None, energy=energy)

    # 简单的聚散约束，避免完全重叠
    for i in range(len(blobs)):
        for j in range(i+1, len(blobs)):
            bi, bj = blobs[i], blobs[j]
            dx, dy = bj.x - bi.x, bj.y - bi.y
            d = math.hypot(dx, dy) + 1e-5
            if d < (bi.r + bj.r) * 1.1:
                push = (bi.r + bj.r) * 1.1 - d
                bi.x -= (dx/d) * push * 0.2
                bi.y -= (dy/d) * push * 0.2
                bj.x += (dx/d) * push * 0.2
                bj.y += (dy/d) * push * 0.2

    # 发光层 + 绿色描边（呼应参考视觉）
    blob_layer = pg.Surface((W, H), pg.SRCALPHA)
    for b in blobs:
        b.draw(blob_layer)
    outline = pg.Surface((W, H), pg.SRCALPHA)
    for b in blobs:
        pg.draw.circle(outline, (40, 255, 120, 90), (int(b.x), int(b.y)), int(b.r*1.12), width=8)
        pg.draw.circle(outline, (20, 180, 90, 120), (int(b.x), int(b.y)), int(b.r*1.24), width=10)
    screen.blit(blob_layer, (0, 0))
    screen.blit(outline, (0, 0))

    # 相机小窗
    pg.draw.rect(screen, (20, 20, 20), CAM_RECT, border_radius=8)
    inner = CAM_RECT.inflate(-4, -4)
    screen.blit(frame_surf, inner.topleft)

    # HUD
    title = big_font.render("Hand-tracked Metaballs", True, (30, 30, 30))
    screen.blit(title, (16, 16))
    hud = [
        "Controls: [S] Save screenshot   [Q] Quit",
        f"Energy: {energy:.2f}   (move faster or open fingers to increase)",
        "Tip: 食指指尖 = 黄色点；白色小方块 = 关键指尖/腕；白色曲线 = 指尖连线；",
        "Gameplay: control the size by stretching and extending fingers",
        "Gameplay: move fingers to collide with other bubbles"
    ]
    # Draw HUD at the bottom of the screen
    line_h = 20
    hud_h = len(hud) * line_h
    y = H - 16 - hud_h
    for line in hud:
        surf = font.render(line, True, (30, 30, 30))
        screen.blit(surf, (16, y))
        y += line_h

    # 在主画布标出吸引点
    if attract_world is not None:
        pg.draw.circle(screen, (255, 210, 0), (int(attract_world[0]), int(attract_world[1])), 8)

    pg.display.flip()

# 释放资源
hands.close()
cap.release()
pg.quit()

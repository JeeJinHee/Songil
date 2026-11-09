"""손길 손끝 유도 테스트 (녹화 영상 / 실시간 웹캠 공용)

녹화 영상:  python guide_sim.py --video 촬영.mp4 --target COFFEE
            → out/guide_<영상이름>.mp4 (디버그 화면 + 유도 소리가 들어간 영상)
실시간:     python guide_sim.py --camera 0 --target COFFEE
            → 창에 디버그 화면이 뜨고 스피커로 안내. r = 버튼 다시 인식, q = 종료

처리 순서
 1) 기준 프레임(손이 없는 프레임)에서 버튼 인식 → 목표 버튼 찾기 (button_v0.py)
 2) 매 프레임: 손 관절 추적(MediaPipe) → 검지 손끝 좌표
 3) 폰 흔들림 보정: 기준 프레임과 현재 프레임의 특징점 매칭으로 목표 버튼 위치 갱신
 4) 안내: "화면에 손가락을 대 주세요" → 방향 음성(오른쪽/왼쪽/위쪽/아래쪽)
          → 가까워질수록 빨라지는 "삐" → 도착을 예측해 "클릭!"
"""
import argparse, os, sys, time, math, json, wave, subprocess, threading, queue, urllib.request
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import button_v0
from button_v0 import analyze, sim   # noqa: E402
MENU = None    # --menu: 메뉴 사전(kiosk_menu.json)으로 OCR 글자 보정 (menu_dict.MenuDict)
PLAN = None    # --server: 다음 목표를 BE에게 물어봄 (be_client.BEPlanner) / --planner: 노트북에서 정함 (planner.LocalPlanner)
BOOST = True   # 사전과 안 맞은 버튼을 잘라서 크게 다시 읽기 (ocr_boost). --no-boost 로 끔
TIP_SHIFT = (0.0, 0.0)   # 손끝 위치 보정(px): check_tips.py eval 이 알려 주는 평균 치우침을 --tip-shift 로 넣음
def btxt(b):   # 비교에 쓸 버튼 글자: 사전 보정을 켰으면 보정된 이름, 아니면 OCR 원문
    return b["name"] if MENU is not None else b.get("raw", b["name"])

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
              (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

# ---------------- 설정값 (실험하면서 조정) ----------------
CFG = dict(
    latency=0.35,        # "클릭!"을 미리 말할 시간(초): 처리 지연 + 사람 반응 시간
    beep_start=6.0,      # 목표까지 거리가 버튼 크기의 몇 배 이내일 때부터 삐 소리
    beep_min=0.08,       # 가장 가까울 때 삐 간격(초)
    beep_max=0.9,        # 가장 멀 때 삐 간격(초)
    dir_gap=1.5,         # 방향 음성 최소 간격(초)
    dir_repeat=3.5,      # 가까워지지 않으면 방향 음성을 다시 말하는 간격(초)
    hold_ready=0.5,      # 손끝이 이 시간 이상 보이면 유도 시작(초)
    predict=0.25,        # 방향 판단에 쓸 '예측 위치' 시간(초): 처리·음성 지연만큼 앞을 내다봄
    dir_hold=0.4,        # 새 방향이 이 시간 이상 유지돼야 말함(초): 경계에서 왔다 갔다 방지
    edge_margin=0.12,    # 버튼 폭/높이의 이 비율만큼 안쪽에 들어오면 그 축은 맞은 것으로 봄
    click_margin=0.0,    # "클릭!" 판정: 손끝이 버튼 테두리에서 (버튼 크기 x 이 값)만큼 안쪽이어야 함. 0=테두리 안, 0.15=더 안쪽 (v0.5까지는 -0.15 = 테두리 밖 15%도 인정)
    click_eta_ratio=1.0, # 손끝이 버튼 크기의 이 배수 이내 + 도착 예상 시간 < latency 이면 미리 "클릭!" (v0.5는 1.5)
)
DIRS = ("오른쪽", "왼쪽", "위쪽", "아래쪽")
CHANGE_THR = 0.15        # 손을 뺀 화면에서 이 비율 이상 바뀌면 '화면 바뀜' (--change-thr 로 조정)

# ---------------- 손 추적 ----------------
class HandTracker:
    """세로로 긴 영상은 MediaPipe가 정사각형으로 줄이면서 손이 작아져 놓치기 쉽다.
    그래서 ① 직전 손 주변을 정사각형으로 잘라 먼저 찾고 ② 없으면 화면을 겹치는 정사각형 타일로 나눠 찾는다."""
    def __init__(self, model_path, conf=0.3):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        import mediapipe as mp
        if not os.path.exists(model_path):
            print("손 모델 다운로드 중...", MODEL_URL)
            urllib.request.urlretrieve(MODEL_URL, model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=conf, min_hand_presence_confidence=conf)
        self.det = vision.HandLandmarker.create_from_options(opts)
        self.mp = mp
        self.last_box = None   # 직전 손 영역 (x0,y0,x1,y1)

    def _run(self, img, ox, oy):
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = self.det.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        h, w = img.shape[:2]
        return [[(p.x * w + ox, p.y * h + oy) for p in lm] for lm in res.hand_landmarks]

    @staticmethod
    def _size(pts):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    def detect(self, bgr, ts_ms=None):
        H, W = bgr.shape[:2]; S = min(H, W)
        hands = []
        if self.last_box is not None:                       # ① 직전 손 주변
            x0, y0, x1, y1 = self.last_box
            side = int(min(S, max(256, 2.4 * max(x1 - x0, y1 - y0))))
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            ox = int(min(max(0, cx - side / 2), W - side)); oy = int(min(max(0, cy - side / 2), H - side))
            hands = self._run(bgr[oy:oy + side, ox:ox + side], ox, oy)
        if not hands:                                        # ② 겹치는 정사각형 타일
            n = max(1, math.ceil((max(H, W) - S) / (S * 0.5)) + 1)
            for k in range(n):
                off = int(round(k * (max(H, W) - S) / max(1, n - 1))) if n > 1 else 0
                if H >= W: hands += self._run(bgr[off:off + S, :], 0, off)
                else: hands += self._run(bgr[:, off:off + S], off, 0)
        if not hands: self.last_box = None; return None
        best = max(hands, key=self._size)                    # 반사 등으로 손이 2개면 큰 손
        xs = [p[0] for p in best]; ys = [p[1] for p in best]
        self.last_box = (min(xs), min(ys), max(xs), max(ys))
        return best

# ---------------- 보완: 피부색으로 손끝 찾기 (MediaPipe가 손을 못 찾을 때만) ----------------
class SkinTip:
    """화면 아래·오른쪽 가장자리에서 들어오는 살색 영역을 찾아, 들어온 쪽에서 가장 먼 점을 손끝으로 본다.
    손등이 프레임 밖으로 잘려 MediaPipe가 손을 못 찾는 장면(손가락 하나만 위로 뻗은 모양)을 위한 보완책."""
    def __init__(self, edges=("bottom", "right")):
        self.edges = edges; self.contour = None
    def find(self, bgr, screen_quad=None):
        """screen_quad: 현재 프레임 기준 키오스크 화면 영역(네 꼭짓점). 벽·포스터의 살색을 피하려고 이 안에서만 찾는다."""
        H, W = bgr.shape[:2]; self.contour = None; self.base = None
        s = 480 / max(H, W); small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb); hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        m = ((ycc[..., 1] >= 135) & (ycc[..., 1] <= 175) & (ycc[..., 2] >= 80) & (ycc[..., 2] <= 130)
             & (hsv[..., 1] >= 30) & (hsv[..., 2] >= 60)).astype(np.uint8) * 255
        h0, w0 = m.shape
        # 키오스크 화면 = 밝고 채도 낮은 가장 큰 영역의 볼록 껍질 (손이 가린 부분까지 포함)
        bright = ((hsv[..., 2] > 150) & (hsv[..., 1] < 60)).astype(np.uint8) * 255
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n_, lab, st_, _ = cv2.connectedComponentsWithStats(bright)
        inner = None
        if n_ > 1:
            i = 1 + np.argmax(st_[1:, cv2.CC_STAT_AREA])
            if st_[i, cv2.CC_STAT_AREA] > 0.15 * h0 * w0:
                cnt = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                inner = np.zeros_like(m); cv2.fillPoly(inner, [cv2.convexHull(max(cnt, key=cv2.contourArea))], 255)
                mg = max(5, int(0.06 * h0))                       # 화면 아래쪽 검은 테두리까지 포함 (옆쪽은 벽과 붙기 쉬워 제외)
                down = np.zeros_like(inner); down[mg:] = inner[:-mg]
                inner = cv2.bitwise_or(inner, down)
                m[inner == 0] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        h, w = m.shape; best = None
        for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            if cv2.contourArea(c) < 0.004 * h * w: continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bh < 0.10 * h and bw < 0.10 * w: continue           # 작은 반사·버튼 조각 제외
            pts = c.reshape(-1, 2)
            edge = np.zeros(len(pts), bool)                        # 영상 가장자리에 닿은 점
            if "bottom" in self.edges: edge |= pts[:, 1] >= h - 3
            if "right" in self.edges: edge |= pts[:, 0] >= w - 3
            if "left" in self.edges: edge |= pts[:, 0] <= 2
            border = np.zeros(len(pts), bool)                      # 화면(+테두리) 영역의 아래쪽 경계에 닿은 점
            if inner is not None:
                er = cv2.erode(inner, np.ones((7, 7), np.uint8))
                yy = np.clip(pts[:, 1], 0, h0 - 1); xx = np.clip(pts[:, 0], 0, w0 - 1)
                ys_, xs_ = np.nonzero(inner); cy, cx = ys_.mean(), xs_.mean(); hw = xs_.max() - xs_.min()
                right_zone = pts[:, 0] > cx + 0.35 * hw if "right" in self.edges else np.zeros(len(pts), bool)
                border = (er[yy, xx] == 0) & ((pts[:, 1] > cy) | right_zone)   # 손은 아래·오른쪽에서 들어온다
            touch = edge if edge.sum() >= 3 else border
            if edge.sum() < 3 and bh < 0.08 * h: continue           # 가장자리에 안 닿으면 세로로 충분히 길어야 손 (키패드 버튼 띠 제외)
            if touch.sum() < 3: continue                          # 가장자리에서 들어오는 영역만
            base = pts[touch].mean(0)
            d = np.linalg.norm(pts - base, axis=1); tip = pts[np.argmax(d)]
            tp = pts[touch]; extent = np.linalg.norm(tp.max(0) - tp.min(0)) + 1
            if d.max() < 0.75 * extent: continue                   # 가장자리를 따라 길게 붙은 띠(벽 등)는 손가락이 아님
            if best is None or d.max() > best[0]: best = (d.max(), tip, c, base)
        if best is None: return None
        self.contour = (best[2].astype(np.float32) / s).astype(np.int32)
        self.base = (float(best[3][0] / s), float(best[3][1] / s))   # 손이 들어온 쪽 (손가락 방향 계산용)
        return (float(best[1][0] / s), float(best[1][1] / s))

# ---------------- 손끝 위치 보정: 추정한 손끝 주변에서 손가락의 실제 끝점으로 당겨 오기 ----------------
def refine_tip(bgr, est, direction, win_ratio=0.06, iters=4):
    """창 가장자리까지 손가락이 이어지면 그 점에서 다시 찾는 것을 반복 (최대 iters번)"""
    cur = est; limit = 0.12 * max(bgr.shape[:2])               # 원래 추정에서 너무 멀리 가지 않게
    for _ in range(iters):
        nxt = _refine_once(bgr, cur, direction, win_ratio)
        if nxt is None or math.dist(nxt, est) > limit: return None if cur is est else cur
        if math.dist(nxt, cur) < 2: return nxt
        cur = nxt
    return cur

def _refine_once(bgr, est, direction, win_ratio):
    """est 주변 창에서 살색 영역을 찾고, 손가락 방향(direction)으로 가장 멀리 나간 점을 손끝으로 쓴다.
    따라가기가 손가락 마디 쪽으로 미끄러지거나 스켈레톤 끝점이 어긋나는 오류를 바로잡는다."""
    H, W = bgr.shape[:2]; r = int(win_ratio * max(H, W))
    x0, y0 = int(max(0, est[0] - r)), int(max(0, est[1] - r)); x1, y1 = int(min(W, est[0] + r)), int(min(H, est[1] + r))
    if x1 - x0 < 10 or y1 - y0 < 10: return None
    crop = bgr[y0:y1, x0:x1]
    ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb); hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = ((ycc[..., 1] >= 133) & (ycc[..., 1] <= 180) & (ycc[..., 2] >= 77) & (ycc[..., 2] <= 135) & (hsv[..., 1] >= 25) & (hsv[..., 1] <= 170)).astype(np.uint8)  # 진한 빨강(선택 테두리 등) 제외
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    ex, ey = est[0] - x0, est[1] - y0
    ys, xs = np.nonzero(lab)
    d2 = (xs - ex) ** 2 + (ys - ey) ** 2; k = np.argmin(d2)
    if d2[k] > (0.25 * r) ** 2: return None                     # 추정 위치 근처에 살색이 없으면 보정하지 않음
    comp = lab == lab[ys[k], xs[k]]
    if comp.sum() < 0.01 * comp.size: return None
    cy, cx = np.nonzero(comp)
    dx, dy = direction; proj = (cx - ex) * dx + (cy - ey) * dy
    top = proj >= np.percentile(proj, 99)
    return (float(cx[top].mean() + x0), float(cy[top].mean() + y0))

def unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return tuple(v / n) if n > 1e-6 else None

# ---------------- 손끝 따라가기: 한 번 찾은 손끝을 광학 흐름으로 다음 프레임에서 추적 ----------------
class TipFlow:
    """손끝 바로 안쪽(손가락 위) 작은 점들을 따라가고, 손끝 = 따라간 점 + 처음의 손끝 오프셋.
    손 전체를 다시 찾지 않으므로 손등이 잘려도 계속 따라갈 수 있다."""
    def __init__(self, max_frames=45):
        self.prev = None; self.pts = None; self.offset = None; self.age = 0; self.max_frames = max_frames
    def start(self, gray, tip, toward):
        """tip: 손끝, toward: 손 안쪽 방향의 한 점(관절 7번 또는 피부 영역의 들어온 쪽)"""
        tip = np.float32(tip); v = np.float32(toward) - tip; n = np.linalg.norm(v)
        if n < 1: return
        v /= n; r = max(8.0, 0.025 * max(gray.shape))
        anchor = tip + v * r * 1.2
        g = np.array([[i, j] for i in (-0.5, 0, 0.5) for j in (-0.5, 0, 0.5)], np.float32) * r
        self.pts = (anchor + g).reshape(-1, 1, 2).astype(np.float32)
        self.offset = tip - anchor; self.prev = gray; self.age = 0
    def track(self, gray):
        if self.pts is None: return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.pts, None, winSize=(21, 21), maxLevel=3)
        p0, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None, winSize=(21, 21), maxLevel=3)
        fb = np.linalg.norm((p0 - self.pts).reshape(-1, 2), axis=1)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.5)
        self.age += 1
        if ok.sum() < 4 or self.age > self.max_frames: self.stop(); return None
        d = np.median((p1 - self.pts).reshape(-1, 2)[ok], axis=0)
        self.pts = (self.pts + d.reshape(1, 1, 2)).astype(np.float32); self.prev = gray
        tip = self.pts.reshape(-1, 2).mean(0) + self.offset
        h, w = gray.shape
        if not (0 <= tip[0] < w and 0 <= tip[1] < h): self.stop(); return None
        return (float(tip[0]), float(tip[1]))
    def stop(self): self.pts = None

# ---------------- 손끝 떨림 줄이기 (One Euro 필터) ----------------
class OneEuro:
    def __init__(self, mincutoff=1.0, beta=0.02, dcutoff=1.0):
        self.mc, self.b, self.dc = mincutoff, beta, dcutoff; self.x = self.dx = self.t = None
    @staticmethod
    def _a(cut, dt): r = 2 * math.pi * cut * dt; return r / (r + 1)
    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self.x is None: self.x, self.dx, self.t = x, np.zeros_like(x), t; return x
        dt = max(1e-3, t - self.t); self.t = t
        dx = (x - self.x) / dt; a = self._a(self.dc, dt); self.dx = a * dx + (1 - a) * self.dx
        cut = self.mc + self.b * np.linalg.norm(self.dx); a = self._a(cut, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x
    def reset(self): self.x = None

# ---------------- 폰 흔들림 보정 ----------------
class ScreenTracker:
    def __init__(self, ref_bgr, hand_pts=None):
        self.orb = cv2.ORB_create(1500)
        g = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        self.kp, self.des = self.orb.detectAndCompute(g, self._mask(g.shape, hand_pts))
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.H = np.eye(3); self.ok = True
    @staticmethod
    def _mask(shape, hand_pts):
        m = np.full(shape, 255, np.uint8)
        if hand_pts:
            xs = [p[0] for p in hand_pts]; ys = [p[1] for p in hand_pts]
            pad = 0.6 * max(max(xs) - min(xs), max(ys) - min(ys))
            cv2.rectangle(m, (int(min(xs) - pad), int(min(ys) - pad)), (shape[1], shape[0]), 0, -1)  # 손~팔(아래·오른쪽) 제외
        return m
    def update(self, bgr, hand_pts):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(g, self._mask(g.shape, hand_pts))
        if des is None or self.des is None or len(kp) < 30: self.ok = False; return self.H
        good = []
        for m in self.bf.knnMatch(self.des, des, k=2):
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance: good.append(m[0])
        if len(good) < 25: self.ok = False; return self.H
        src = np.float32([self.kp[m.queryIdx].pt for m in good]); dst = np.float32([kp[m.trainIdx].pt for m in good])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or inl.sum() < 20: self.ok = False; return self.H
        self.H, self.ok = H, True
        return H
    def warp_rect(self, r):
        pts = np.float32([[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]]).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

# ---------------- 안내 로직 ----------------
class Guide:
    """상태: INTRO → WAIT → GUIDE → DONE. 이벤트 (시각, 종류, 문구) 를 쌓는다."""
    def __init__(self, cfg=CFG):
        self.c = cfg; self.state = "INTRO"; self.events = []
        self.ready_since = None; self.last_dir = None; self.last_dir_t = -9; self.best_dist = 1e9
        self.last_beep = -9; self.prev = None; self.speed = 0.0; self.info = {}
        self.vel = np.zeros(2); self.prev_tip = None; self.cand = None; self.cand_t = 0
        self.intro = "화면에 손가락을 대 주세요"
    def say(self, t, text): self.events.append((t, "say", text))
    def step(self, t, tip, quad):
        """tip: 손끝 (x,y) 또는 None, quad: 목표 버튼 네 꼭짓점(현재 영상 기준)"""
        c = self.c
        if self.state == "INTRO":
            self.say(t, self.intro); self.state = "WAIT"; return
        if self.state == "DONE": return
        if tip is None or quad is None:
            self.ready_since = None; self.prev = None; self.prev_tip = None; self.vel[:] = 0
            self.info = {"status": "손끝 없음"}; return
        if self.state == "WAIT":
            self.ready_since = self.ready_since or t
            if t - self.ready_since >= c["hold_ready"]: self.state = "GUIDE"
            self.info = {"status": "손끝 확인 중"}; return
        # --- GUIDE ---
        cx, cy = quad.mean(0)
        bw = np.linalg.norm(quad[1] - quad[0]); bh = np.linalg.norm(quad[3] - quad[0])
        size = max(20.0, min(bw, bh))
        dx, dy = cx - tip[0], cy - tip[1]; dist = math.hypot(dx, dy); ratio = dist / size
        inside = cv2.pointPolygonTest(quad.astype(np.float32), (float(tip[0]), float(tip[1])), True)
        # 다가오는 속도 (px/s, 지수 평활)
        if self.prev is not None:
            dt = max(1e-3, t - self.prev[0]); v = (self.prev[1] - dist) / dt
            self.speed = 0.7 * self.speed + 0.3 * v
        self.prev = (t, dist)
        eta = dist / self.speed if self.speed > 1 else float("inf")
        # 손끝 이동 속도(벡터) → 지연 시간만큼 앞의 예측 위치로 방향을 판단
        if self.prev_tip is not None:
            dt = max(1e-3, t - self.prev_tip[0])
            v = (np.array(tip, float) - self.prev_tip[1]) / dt
            self.vel = 0.6 * self.vel + 0.4 * np.clip(v, -3 * size / 0.2, 3 * size / 0.2)
        self.prev_tip = (t, np.array(tip, float))
        px, py = np.array(tip, float) + self.vel * c["predict"]
        self.info = {"status": "유도 중", "dist": dist, "ratio": ratio, "eta": eta, "dx": dx, "dy": dy, "pred": (px, py)}
        # 도착 예측 → "클릭!"
        if inside > c["click_margin"] * size or (ratio < c["click_eta_ratio"] and eta < c["latency"]):
            self.say(t, "클릭!"); self.state = "DONE"; self.info["status"] = "도착"; return
        # 방향 음성: 버튼 사각형 기준. 예측 위치가 버튼의 가로 범위 안이면 가로는 맞은 것 → 세로만 안내
        x0, y0 = quad.min(0); x1, y1 = quad.max(0)
        mx, my = c["edge_margin"] * (x1 - x0), c["edge_margin"] * (y1 - y0)
        need_x = not (x0 + mx <= px <= x1 - mx); need_y = not (y0 + my <= py <= y1 - my)
        # 둘 다 어긋나면 한 번에 한 축만: 가로를 먼저 맞추고 세로로 (헷갈리지 않게)
        if need_x: d = "오른쪽" if px < x0 + mx else "왼쪽"
        elif need_y: d = "아래쪽" if py < y0 + my else "위쪽"
        else: d = self.last_dir                      # 예측 위치가 이미 버튼 안: 방향은 더 말하지 않음
        # 방향이 바뀌면 dir_hold 동안 유지될 때만 인정 (경계에서 왔다 갔다 방지)
        if d != self.cand: self.cand, self.cand_t = d, t
        stable = d is not None and (t - self.cand_t >= c["dir_hold"] or self.last_dir is None)
        improving = dist < self.best_dist - 0.3 * size
        if improving: self.best_dist = dist
        if stable and ((d != self.last_dir and t - self.last_dir_t >= c["dir_gap"]) or
                       (d == self.last_dir and t - self.last_dir_t >= c["dir_repeat"] and not improving)):
            self.say(t, d); self.last_dir, self.last_dir_t = d, t; self.best_dist = dist
        # 후진 경고음처럼: 가까울수록 빠른 삐
        if ratio < c["beep_start"]:
            k = max(0.0, min(1.0, (ratio - 1.0) / (c["beep_start"] - 1.0)))
            interval = c["beep_min"] + (c["beep_max"] - c["beep_min"]) * k
            if t - self.last_beep >= interval:
                self.events.append((t, "beep", "")); self.last_beep = t
        self.info["dir"] = d

# ---------------- 소리 ----------------
SR = 22050
def beep_wave(freq=1600, dur=0.06):
    t = np.arange(int(SR * dur)) / SR; env = np.minimum(1, np.minimum(t, dur - t) / 0.005)
    return (0.5 * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)

def tts_wave(text, cache_dir):
    """pyttsx3(윈도우/맥 기본 음성) → macOS say → 실패 시 None(자막만 표시)"""
    os.makedirs(cache_dir, exist_ok=True)
    fn = os.path.join(cache_dir, f"{abs(hash(text))}.wav")
    if not os.path.exists(fn):
        try:
            import pyttsx3
            eng = pyttsx3.init()
            for v in eng.getProperty("voices"):
                if "ko" in (str(v.languages) + v.id + v.name).lower() or "korean" in v.name.lower(): eng.setProperty("voice", v.id); break
            eng.setProperty("rate", 190); eng.save_to_file(text, fn); eng.runAndWait()
        except Exception:
            try: subprocess.run(["say", "-v", "Yuna", "-o", fn, "--data-format=LEI16@22050", text], check=True, capture_output=True)
            except Exception: return None
    try:
        import soundfile as sf
        data, sr = sf.read(fn, dtype="float32")
    except Exception:
        try:
            with wave.open(fn) as w:
                sr = w.getframerate(); data = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
                if w.getnchannels() == 2: data = data.reshape(-1, 2).mean(1)
        except Exception: return None
    if data.ndim > 1: data = data.mean(1)
    if sr != SR: data = np.interp(np.linspace(0, len(data), int(len(data) * SR / sr), endpoint=False), np.arange(len(data)), data).astype(np.float32)
    return data

def render_audio(events, duration, path, cache_dir):
    out = np.zeros(int(SR * (duration + 3)), np.float32); b = beep_wave(); voiced = True
    for t, kind, text in events:
        w = b if kind == "beep" else tts_wave(text, cache_dir)
        if w is None: voiced = False; continue
        i = int(t * SR); out[i:i + len(w)] += w[: max(0, len(out) - i)]
    out = np.clip(out, -1, 1)
    with wave.open(path, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes((out * 32767).astype(np.int16).tobytes())
    return voiced

class LivePlayer:
    """실시간 모드: 삐는 바로, 음성은 별도 스레드에서 재생"""
    def __init__(self, cache_dir):
        self.q = queue.Queue(); self.cache = cache_dir
        try: import sounddevice as sd; self.sd = sd
        except Exception: self.sd = None; print("sounddevice가 없어 소리 없이 진행합니다 (pip install sounddevice)")
        self.beep = beep_wave(); threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        while True:
            kind, text = self.q.get()
            if self.sd is None: continue
            w = self.beep if kind == "beep" else tts_wave(text, self.cache)
            if w is None: continue
            self.cur = text if kind == "say" else None
            self.sd.play(w, SR); self.sd.wait(); self.cur = None
    def push(self, kind, text):
        if kind == "beep" and not self.q.empty(): return   # 밀린 삐는 버림
        if kind == "say":
            # 새 안내가 오면 아직 못 한 안내는 버림 (늦게 나오는 엉뚱한 방향 방지)
            while not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: break
            # 지금 말하는 게 방향 음성이고 새 안내가 다르면 끊고 바로 새 안내
            if self.sd is not None and getattr(self, "cur", None) in DIRS and text != self.cur:
                try: self.sd.stop()
                except Exception: pass
        self.q.put((kind, text))

# ---------------- 그리기 ----------------
FONT = None
def put_kr(img, text, xy, size=28, color=(255, 255, 255), bg=(0, 0, 0)):
    global FONT
    from PIL import Image, ImageDraw, ImageFont
    if FONT is None:
        for p in ["C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
            if os.path.exists(p): FONT = p; break
    f = ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(im)
    l, t_, r, b = d.textbbox(xy, text, font=f)
    if bg is not None: d.rectangle([l - 8, t_ - 6, r + 8, b + 6], fill=bg[::-1])
    d.text(xy, text, font=f, fill=color[::-1])
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def draw(frame, buttons_quads, target_quad, hand, tip, guide, subtitle, beeping, tracking_ok, method="", skin_contour=None):
    vis = frame.copy(); s = max(1, frame.shape[1] // 600)
    for q in buttons_quads: cv2.polylines(vis, [q.astype(int)], True, (200, 200, 200), s)
    if target_quad is not None: cv2.polylines(vis, [target_quad.astype(int)], True, (0, 0, 255), 4 * s)
    if hand:
        for a, b in HAND_EDGES: cv2.line(vis, tuple(map(int, hand[a])), tuple(map(int, hand[b])), (0, 220, 0), 2 * s)
        for p in hand: cv2.circle(vis, tuple(map(int, p)), 3 * s, (0, 160, 0), -1)
    if skin_contour is not None: cv2.drawContours(vis, [skin_contour], -1, (255, 255, 0), 2 * s)
    if tip is not None:
        cv2.circle(vis, tuple(map(int, tip)), 9 * s, (0, 0, 255), -1)
        pred = guide.info.get("pred")
        if pred is not None and guide.state == "GUIDE":
            cv2.circle(vis, tuple(map(int, pred)), 7 * s, (255, 0, 255), 2 * s)   # 보라 원 = 예측 위치
        if target_quad is not None and guide.state == "GUIDE":
            cv2.arrowedLine(vis, tuple(map(int, tip)), tuple(map(int, target_quad.mean(0))), (0, 165, 255), 3 * s, tipLength=0.08)
    i = guide.info
    line = f"상태: {i.get('status', guide.state)}"
    if "ratio" in i: line += f" | 거리: 버튼 {i['ratio']:.1f}개 | 도착 예상 {min(i['eta'], 99):.1f}초"
    if method: line += f" | 손끝: {method}"
    if not tracking_ok: line += " | 흔들림 보정 실패"
    vis = put_kr(vis, line, (20, 20), 26 * s // 2 + 10)
    if beeping: vis = put_kr(vis, "삐", (frame.shape[1] - 90, 20), 40, (0, 0, 0), (0, 220, 255))
    if subtitle: vis = put_kr(vis, f"안내: {subtitle}", (20, frame.shape[0] - 90), 40, (255, 255, 255), (0, 0, 160))
    return vis

# ---------------- 기준 프레임에서 버튼 인식 ----------------
def recognize(frame, target, tmp_png):
    cv2.imwrite(tmp_png, frame)
    _, res = analyze(tmp_png)
    btns = res["buttons"]
    if MENU is not None:
        MENU.correct(btns)
        if BOOST:
            import ocr_boost; ocr_boost.boost(frame, btns, MENU)
    if PLAN is not None:              # 서버 모드: 목표는 서버가 정함 (buttonId로 받음)
        best = PLAN.decide(btns, MENU.cart_rows(res) if MENU is not None else [])
        return best, btns, res
    if not btns: return None, [], res
    from button_v0 import jamo
    def key(b):                       # 유사도가 같으면 글자 수가 목표와 비슷한 버튼을 우선 ('확인' vs '확인 후 결제')
        txt = btxt(b); return (sim(target, txt), -abs(len(jamo(txt)) - len(jamo(target))))
    best = max(btns, key=key)
    score = sim(target, btxt(best))
    # 짧은 목표('확인'·'결제' 등)는 한 글자만 겹쳐도 점수가 0.5가 나와서('확인' vs '확장') 기준을 높임
    need = 0.6 if len(jamo(target)) <= 8 else 0.5
    print(f"목표 '{target}' → {best['id']} '{best['name']}' (유사도 {score:.2f}, 기준 {need})")
    if score < need:
        print("목표 버튼을 찾지 못했습니다. 후보:", [b["name"] for b in btns][:15]); return None, btns, res
    return best, btns, res

def save_screen(outdir, idx, frame, res, target_name, found):
    """인식한 화면 원본 + 버튼 목록 저장 (BE DB 정리용·정확도 확인용)"""
    d = os.path.join(outdir, "screens"); os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"{idx:02d}_{''.join(ch for ch in target_name if ch.isalnum())}")
    cv2.imwrite(stem + ".png", frame)
    h, w = frame.shape[:2]
    btns = [dict(id=b["id"], name=b["name"], bbox=[round(b["rect"][0] / w, 4), round(b["rect"][1] / h, 4),
                 round((b["rect"][2] - b["rect"][0]) / w, 4), round((b["rect"][3] - b["rect"][1]) / h, 4)],
                 conf=b.get("conf"), selected=b.get("selected", False),
                 **({"raw": b["raw"]} if "raw" in b else {}), **({"group": b["group"]} if b.get("group") else {})) for b in res.get("buttons", [])]
    btns.sort(key=lambda b: (round(b["bbox"][1], 2), b["bbox"][0]))          # 위→아래, 왼쪽→오른쪽
    json.dump(dict(target=target_name, found=found["name"] if found else None, image_w=w, image_h=h,
                   buttons=btns, info_texts=res.get("info_texts", [])),
              open(stem + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return stem

def open_camera(idx, backend="auto"):
    """윈도우에서는 DirectShow(dshow)로 먼저 열기: DroidCam 같은 가상 카메라가 MSMF로 열면 초록 화면만 나오는 경우가 있음"""
    order = {"dshow": [cv2.CAP_DSHOW], "msmf": [cv2.CAP_MSMF], "any": [cv2.CAP_ANY]}.get(backend)
    if order is None: order = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
    for api in order:
        cap = cv2.VideoCapture(idx, api)
        if cap.isOpened(): return cap
    return cv2.VideoCapture(idx)

def blank_frame(fr):
    """한 가지 색으로 꽉 찬 프레임(가상 카메라 초록 화면·검은 화면)인지"""
    if fr is None: return True
    small = cv2.resize(fr, (64, 36)).reshape(-1, fr.shape[2] if fr.ndim == 3 else 1); return float(small.std(0).max()) < 6.0

def look_ahead(btns, targets, ti, n=2, thr=0.6):
    """지금 목표가 없을 때, 다음 목표(1~2개 뒤)가 화면에 있으면 → 이미 다음 화면으로 넘어간 것"""
    for k in range(ti + 1, min(ti + 1 + n, len(targets))):
        if not btns: break
        best = max(btns, key=lambda b: sim(targets[k], btxt(b)))
        if sim(targets[k], btxt(best)) >= thr: return k, best
    return None, None

def rect_quad(r): return np.float32([[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]])

# ---------------- 실행 ----------------
# ---------------- 화면 바뀜 감지 ----------------
class ScreenWatcher:
    """기준 화면(ref)을 현재 카메라 위치로 맞춘 뒤, 손 영역을 뺀 화면 부분이 얼마나 달라졌는지 본다.
    바뀜이 0.3초 이어지면 '바뀜', 그 뒤 0.5초 동안 화면이 멈추면 '안정' → 새 기준 화면으로 다시 인식."""
    def __init__(self, ref, roi, scale=0.25):
        self.s = scale; self.roi = roi
        self.ref = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale), (5, 5), 0)
        self.prev = None; self.changed_since = None; self.still_since = None; self.last = 0.0
    def score(self, gray, H, hand_pts):
        s = self.s; cur = cv2.GaussianBlur(cv2.resize(gray, None, fx=s, fy=s), (5, 5), 0)
        S = np.diag([s, s, 1.0]); Hs = S @ H @ np.linalg.inv(S)
        h, w = cur.shape
        warped = cv2.warpPerspective(self.ref, Hs, (w, h))
        x0, y0, x1, y1 = self.roi
        q = cv2.perspectiveTransform(np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2), S @ H).reshape(-1, 2)
        m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [q.astype(np.int32)], 255)
        if hand_pts:                                             # 손·팔 영역 제외
            xs = [p[0] * s for p in hand_pts]; ys = [p[1] * s for p in hand_pts]
            pad = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys), 10)
            # 손 주변 + 손 아래(팔이 들어오는 쪽)만 뺌. 예전엔 손에서 오른쪽 아래 끝까지 통째로 빼서, 손을 잘못 잡으면 화면 대부분이 빠졌음
            cv2.rectangle(m, (int(min(xs) - pad), int(min(ys) - pad)), (int(max(xs) + pad), h), 0, -1)
        valid = (m > 0) & (warped > 0)
        if valid.sum() < 500: return 0.0, cur
        diff = cv2.absdiff(cur, warped)
        return float((diff[valid] > 40).mean()), cur
    def update(self, t, gray, H, hand_pts):
        """반환: None / 'changed'(바뀌는 중) / 'stable'(바뀐 뒤 멈춤 → 다시 인식할 때)"""
        sc, cur = self.score(gray, H, hand_pts); self.last = sc
        # '아직 바뀌는 중'인지: 손에 든 폰의 흔들림까지 움직임으로 보지 않도록, 변화 비율이 계속 출렁이는지로 판단
        moving = self.prev is not None and abs(sc - self.prev) > 0.04
        self.prev = sc
        if sc > CHANGE_THR:
            self.changed_since = self.changed_since or t
        elif self.still_since is None:
            self.changed_since = None
        if self.changed_since is not None and t - self.changed_since >= 0.3:
            if moving: self.still_since = None; return "changed"
            self.still_since = self.still_since or t
            if t - self.still_since >= 0.5: return "stable"
            return "changed"
        return None

class Recognizer:
    """OCR(10~20초)을 실시간 모드에서는 별도 스레드로 돌려 화면이 멈추지 않게 한다."""
    def __init__(self, outdir, background):
        self.outdir = outdir; self.bg = background; self.result = None; self.busy = False
    def start(self, frame, target):
        self.busy = True; self.result = None
        def job():
            r = recognize(frame, target, os.path.join(self.outdir, f"_ref_{int(time.time()*1000)%100000}.png"))
            name = r[0]["name"] if (PLAN is not None and r[0] is not None) else target
            self.result = (frame, name) + r; self.busy = False
        if self.bg: threading.Thread(target=job, daemon=True).start()
        else: job()
    def take(self):
        r = self.result; self.result = None; return r

# ---------------- 실행 ----------------
def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    cache = os.path.join(args.outdir, "tts_cache")
    live = args.camera is not None
    cap = open_camera(args.camera, getattr(args, "backend", "auto")) if live else cv2.VideoCapture(args.video)
    if not cap.isOpened(): sys.exit("영상을 열 수 없습니다")
    if live:                                   # 글자 인식 때문에 카메라 해상도를 최대한 높게 요청
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, getattr(args, "width", 1920)); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, getattr(args, "height", 1080))
        print(f"카메라 해상도: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
              " (1280x720보다 낮으면 글자 인식이 잘 안 될 수 있어요)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    class _NoHand:                      # --no-mp: MediaPipe 없이 피부색 + 따라가기만 사용
        def detect(self, *a, **k): return None
    no_mp = getattr(args, "no_mp", False)
    if no_mp and getattr(args, "no_skin", False): sys.exit("--no-mp 와 --no-skin 을 같이 쓰면 손끝을 찾을 방법이 없습니다")
    tracker = args.tracker or (_NoHand() if no_mp else HandTracker(os.path.join(HERE, "hand_landmarker.task"), getattr(args, "hand_conf", 0.3)))
    _skin_ref = SkinTip()
    if args.start: cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    if PLAN is not None: targets = ["(서버가 정함)"]
    else: targets = [x.strip() for x in (getattr(args, "targets", None) or args.target).split(",") if x.strip()]
    ti = 0
    auto = not getattr(args, "no_auto", False)

    # 1) 첫 기준 프레임: 녹화 영상은 --ref 시각 또는 손이 처음 안 보이는 프레임, 실시간은 시작 직후
    ref = None; t0 = time.time()
    while ref is None:
        ok, fr = cap.read()
        if not ok: sys.exit("기준 프레임을 찾지 못했습니다")
        t = (time.time() - t0) if live else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if args.ref is not None and not live and t < args.ref: continue
        if args.ref is None and not live:
            has_hand = (_skin_ref.find(fr) is not None) if no_mp else (tracker.detect(fr, t * 1000) is not None)
            if has_hand: continue
        if live and t < 1.0: continue
        if live and blank_frame(fr):                      # 초록·검은 화면이면 실제 영상이 들어올 때까지 기다림
            if t > 8.0:
                sys.exit("카메라에서 초록(또는 단색) 화면만 들어옵니다.\n"
                         " 1) DroidCam 프로그램에서 해상도를 확인하고 같은 값으로 실행: --width 1280 --height 720\n"
                         " 2) 카메라 여는 방식 바꾸기: --backend msmf (또는 --backend any)\n"
                         " 3) --list-cams 로 다른 번호도 확인")
            continue
        ref = fr
    print(f"버튼 인식 중... (OCR, 10~20초)  목표 순서: {targets}")
    target, btns, res = recognize(ref, targets[ti], os.path.join(args.outdir, "_ref.png"))
    if PLAN is not None and target is not None: targets[ti] = target["name"]
    tries = 1
    while target is None and not live and args.ref is None and tries < 6:   # 녹화 영상: 2초씩 뒤로 가며 목표가 보이는 프레임을 자동으로 찾음
        cap.set(cv2.CAP_PROP_POS_MSEC, (t + 2.0) * 1000); ok, fr = cap.read()
        if not ok: break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if args.end and t > args.end: break
        print(f"  → {t:.1f}초 화면으로 다시 시도 ({tries}/5)"); ref = fr
        target, btns, res = recognize(ref, targets[ti], os.path.join(args.outdir, "_ref.png")); tries += 1
        if PLAN is not None and target is not None: targets[ti] = target["name"]
    n_screen = 0
    if live: save_screen(args.outdir, n_screen, ref, res, targets[ti], target); n_screen += 1
    if target is None and live:
        print("첫 화면에서 목표 버튼을 찾지 못했습니다. 화면이 잘 보이게 맞춘 뒤 r 키를 누르세요")
    elif target is None: sys.exit("목표 버튼을 찾지 못했습니다. --ref 로 목표가 잘 보이고 손이 없는 순간(초)을 지정해 보세요")
    print(f"기준 화면: {t:.1f}초 (안내는 이 시각부터 시작)" if not live else "기준 화면 캡처 완료")
    scr = ScreenTracker(ref); watcher = ScreenWatcher(ref, res["screen_roi"])
    raw_writer = None; marks = ["t,target,recognized,mark"]; n_cap = 0
    guide = Guide(); filt = OneEuro()
    if PLAN is not None and PLAN.message and target is not None: guide.intro = PLAN.message + ". 화면에 손가락을 대 주세요"
    skin = None if getattr(args, "no_skin", False) else SkinTip(); last_method = ""; method_count = {}
    flow = TipFlow(); finger_dir = None; corrected = 0; tip_log = ["t,x,y,method"]
    refine = not getattr(args, "no_refine", False)
    player = LivePlayer(cache) if live else None
    rec = Recognizer(args.outdir, background=live)
    events = []                                   # 모든 안내 (목표가 바뀌어도 이어서 기록)
    phase = "GUIDE" if target is not None else ("RETRY" if live else "END"); t_retry = 0; miss_said = False
    t_click = None; t_phase = 0; recog_reason = ""; hand_off_since = None
    writer = None; n = 0; sub, sub_t, beep_t = "", -9, -9; start_t = None

    def say(t, text):
        nonlocal sub, sub_t
        events.append((t, "say", text)); sub, sub_t = text, t
        if player: player.push("say", text)
    last_server_msg = [""]
    def next_text(name):                     # 다음 목표 안내 문장 (계획/서버 모드면 "COFFEE 탭을 누르세요" 같은 문장)
        if PLAN is not None and PLAN.message: return PLAN.message
        return f"다음은 {name} 버튼이에요"
    def miss_text(name):                     # 목표를 못 찾았을 때 읽어 줄 말 (서버 모드면 서버 메시지)
        if PLAN is not None: return PLAN.message or "다음 버튼을 찾지 못했어요"
        return f"{name} 버튼을 찾지 못했어요"
    def server_done(t):                      # 서버가 DONE이라고 하면 끝
        if PLAN is not None and PLAN.done:
            say(t, PLAN.message or "주문 목표를 다 담았어요"); return True
        return False

    while True:
        ok, fr = cap.read()
        if not ok: break
        t = (time.time() - t0) if live else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if args.end and not live and t > args.end: break
        if start_t is None: start_t = t
        if live and not getattr(args, "no_record", False):           # 카메라 원본 저장 → 나중에 --video 로 다시 돌려 볼 수 있음
            if raw_writer is None:
                raw_writer = cv2.VideoWriter(os.path.join(args.outdir, "live_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (fr.shape[1], fr.shape[0]))
            raw_writer.write(fr)
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        tracked = flow.track(gray)
        hand, raw, method, skin_c = None, None, "", None
        if tracked is None or n % 5 == 0:                       # 따라가기 실패 또는 5프레임마다 다시 찾기
            hand = tracker.detect(fr, t * 1000)
            if hand:
                raw, method = hand[8], "스켈레톤"; flow.start(gray, hand[8], hand[6])
                finger_dir = unit(np.subtract(hand[8], hand[6])) or finger_dir
            elif skin:
                found = skin.find(fr)
                if found is not None:
                    raw, method, skin_c = found, "피부색", skin.contour
                    finger_dir = unit(np.subtract(found, skin.base)) or finger_dir
                    if tracked is None or math.dist(found, tracked) > 40: flow.start(gray, found, skin.base)
        if raw is None and tracked is not None: raw, method = tracked, "따라가기"
        if raw is not None and finger_dir is not None and refine:     # 손끝을 실제 손가락 끝점으로 보정
            fixed = refine_tip(fr, raw, finger_dir)
            if fixed is not None:
                if math.dist(fixed, raw) > 6:
                    corrected += 1
                    flow.start(gray, fixed, (fixed[0] - finger_dir[0] * 40, fixed[1] - finger_dir[1] * 40))
                raw = fixed
        if raw is not None and TIP_SHIFT != (0.0, 0.0): raw = (raw[0] + TIP_SHIFT[0], raw[1] + TIP_SHIFT[1])
        mask_pts = hand or ([tuple(p) for p in skin_c.reshape(-1, 2)] if skin_c is not None else None)
        if raw is not None and not mask_pts: mask_pts = [(raw[0] - 40, raw[1] - 40), (raw[0] + 40, raw[1] + 40)]
        if n % 3 == 0: scr.update(fr, mask_pts)
        quad = scr.warp_rect(target["rect"]) if (scr.ok and target is not None) else None
        tip = None
        if raw is not None:
            if method != last_method and "따라가기" not in (method, last_method): filt.reset()
            tip = filt(raw, t)
        else: filt.reset()
        last_method = method
        if method: method_count[method] = method_count.get(method, 0) + 1
        tip_log.append(f"{t:.3f},{'' if tip is None else round(tip[0],1)},{'' if tip is None else round(tip[1],1)},{method}")

        # ---- 안내 ----
        if phase == "GUIDE":
            k = len(guide.events); guide.step(t, tip, quad)
            for ev in guide.events[k:]:
                events.append(ev)
                if ev[1] == "say": sub, sub_t = ev[2], ev[0]
                else: beep_t = ev[0]
                if player: player.push(ev[1], ev[2])
            if guide.state == "DONE": phase, t_click = "WAIT_CHANGE", t

        # ---- 화면 바뀜 감지 → 새 기준 화면 자동 캡처 → 다시 인식 ----
        if auto and phase in ("GUIDE", "WAIT_CHANGE") and n % 2 == 0:
            st = watcher.update(t, gray, scr.H, mask_pts)
            if st == "stable":
                expected = phase == "WAIT_CHANGE"
                if expected: ti += 1
                if PLAN is not None:
                    if expected: PLAN.clicked(True)          # 누른 뒤 화면이 바뀜 → 다음 인식 때 selection-result로 보냄
                    if ti >= len(targets): targets.append("(서버가 정함)")
                if ti >= len(targets):
                    say(t, "모든 단계를 마쳤어요"); phase = "END"
                else:
                    if not expected: say(t, "화면이 바뀌었어요. 다시 확인할게요")
                    recog_reason = "기대한 화면" if expected else "예상 밖 화면"
                    rec.start(fr.copy(), targets[ti]); phase, t_phase = "RECOG", t
                    print(f"[{t:.1f}s] 화면 바뀜({recog_reason}) → '{targets[ti]}' 인식 시작")
            elif phase == "WAIT_CHANGE" and st is None and t - t_click > 4.0:
                # 화면 바뀜을 못 잡았을 수도 있으니, 지금 화면을 다시 찍어 확인한 뒤 결정 (VERIFY)
                if not rec.busy:
                    if PLAN is not None: PLAN.clicked(False)
                    rec.start(fr.copy(), targets[ti]); phase = "VERIFY"
        if phase == "RECOG" and not rec.busy:
            r = rec.take()
            if r is not None:
                rframe, rtarget, tgt, b2, res2 = r
                if PLAN is not None: targets[ti] = rtarget
                if live: save_screen(args.outdir, n_screen, rframe, res2, rtarget, tgt); n_screen += 1
                if server_done(t): phase = "END"
                elif tgt is not None:
                    target, btns, res = tgt, b2, res2
                    scr = ScreenTracker(rframe, mask_pts); watcher = ScreenWatcher(rframe, res["screen_roi"])
                    guide = Guide(); guide.state = "WAIT"; say(t, next_text(rtarget)); phase = "GUIDE"
                else:
                    target, btns, res = None, b2, res2           # 예전 목표 상자가 남아 헷갈리지 않게 지움
                    scr = ScreenTracker(rframe); watcher = ScreenWatcher(rframe, res["screen_roi"])
                    if PLAN is not None and PLAN.message:
                        say(t, PLAN.message); last_server_msg[0] = PLAN.message
                        miss_said = True; phase, t_retry = "RETRY", t + 2.0
                    else:
                        say(t, "손을 잠시 화면에서 떼 주세요"); phase, hand_off_since = "HAND_OFF", None
        if phase == "HAND_OFF":                                  # 손이 0.5초 안 보이면 다시 촬영해 인식
            if raw is None: hand_off_since = hand_off_since or t
            else: hand_off_since = None
            if hand_off_since and t - hand_off_since >= 0.5:
                rec.start(fr.copy(), targets[ti]); phase = "RECOG2"
        if phase == "RECOG2" and not rec.busy:
            r = rec.take()
            if r is not None:
                rframe, rtarget, tgt, b2, res2 = r
                if PLAN is not None: targets[ti] = rtarget
                if PLAN is not None and PLAN.message != last_server_msg[0]: miss_said = False   # 서버가 새 말을 하면 다시 읽어 줌
                if live and (tgt is not None or not miss_said): save_screen(args.outdir, n_screen, rframe, res2, rtarget, tgt); n_screen += 1
                if server_done(t): phase = "END"
                elif tgt is not None:
                    target, btns, res = tgt, b2, res2
                    scr = ScreenTracker(rframe); watcher = ScreenWatcher(rframe, res["screen_roi"])
                    guide = Guide(); say(t, next_text(rtarget) + ". 화면에 손가락을 대 주세요"); guide.state = "WAIT"; phase = "GUIDE"; miss_said = False
                else:
                    target, btns, res = None, b2, res2
                    scr = ScreenTracker(rframe); watcher = ScreenWatcher(rframe, res["screen_roi"])
                    if live:                             # 실시간: 2초마다 자동으로 다시 찍어 인식 (안내는 처음 한 번만)
                        if not miss_said:
                            say(t, miss_text(rtarget) + ("" if PLAN is not None else ". 화면을 비춰 주세요")); miss_said = True
                            if PLAN is not None: last_server_msg[0] = PLAN.message
                        phase, t_retry = "RETRY", t + 2.0
                        print(f"[{t:.1f}s] '{rtarget}' 못 찾음 → 2초 뒤 다시 인식")
                    else:
                        say(t, miss_text(rtarget)); phase = "END"
        if phase == "VERIFY" and not rec.busy:
            r = rec.take()
            if r is not None:
                rframe, rtarget, tgt, b2, res2 = r
                # "클릭!" 뒤에만 쓰는 확인: 방금 누른 목표가 화면에 그대로면 안 눌린 것, 사라지고 바로 다음 목표가 보이면 눌린 것
                k, nb = (None, None) if (tgt is not None or PLAN is not None) else look_ahead(b2, targets, ti, n=1)
                if PLAN is not None: targets[ti] = rtarget
                if server_done(t): phase = "END"
                elif PLAN is not None and tgt is not None and not PLAN.repeat:   # 서버가 새 목표를 줌 → 눌린 것
                    target, btns, res = tgt, b2, res2
                    scr = ScreenTracker(rframe, mask_pts); watcher = ScreenWatcher(rframe, res["screen_roi"])
                    if live: save_screen(args.outdir, n_screen, rframe, res2, rtarget, tgt); n_screen += 1
                    guide = Guide(); guide.state = "WAIT"; say(t, next_text(rtarget)); phase = "GUIDE"
                elif tgt is not None:
                    target, btns, res = tgt, b2, res2
                    scr = ScreenTracker(rframe, mask_pts); watcher = ScreenWatcher(rframe, res["screen_roi"])
                    say(t, "다시 한번 눌러 주세요"); guide = Guide(); guide.state = "WAIT"; phase = "GUIDE"
                elif k is not None:
                    print(f"[{t:.1f}s] '{rtarget}'은 눌린 것으로 판단 (다음 목표 '{targets[k]}'가 보임)")
                    if live: save_screen(args.outdir, n_screen, rframe, res2, targets[k], nb); n_screen += 1
                    ti, target, btns, res = k, nb, b2, res2
                    scr = ScreenTracker(rframe, mask_pts); watcher = ScreenWatcher(rframe, res["screen_roi"])
                    guide = Guide(); guide.state = "WAIT"; say(t, f"다음은 {targets[k]} 버튼이에요"); phase = "GUIDE"
                else:                                            # 둘 다 안 보임 → 손을 치우게 하고 다시 인식
                    say(t, "손을 잠시 화면에서 떼 주세요"); phase, hand_off_since = "HAND_OFF", None
        if phase == "RETRY" and t >= t_retry and not rec.busy:
            rec.start(fr.copy(), targets[min(ti, len(targets) - 1)]); phase = "RECOG2"

        status_extra = {"RECOG": "새 화면 인식 중", "RECOG2": "새 화면 인식 중", "HAND_OFF": "손 떼기 기다리는 중",
                        "WAIT_CHANGE": "화면 바뀜 기다리는 중", "RETRY": "목표 버튼 찾는 중 (화면을 비춰 주세요)", "VERIFY": "눌렸는지 확인 중", "END": "끝"}.get(phase)
        if status_extra: guide.info = {"status": status_extra}
        vis = draw(fr, [scr.warp_rect(b["rect"]) for b in btns], quad, hand, tip, guide,
                   sub if t - sub_t < 1.5 else "", t - beep_t < 0.06, scr.ok, method, skin_c)
        vis = put_kr(vis, (f"목표(자동) {ti + 1}: " if PLAN is not None else f"목표 {ti + 1}/{len(targets)}: ") + f"{targets[min(ti, len(targets) - 1)]}"
                          f" | 화면 변화 {watcher.last * 100:.0f}% (기준 {CHANGE_THR * 100:.0f}%)", (20, 70), 28)
        if live:
            cv2.imshow("songil guide", cv2.resize(vis, None, fx=args.scale, fy=args.scale))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("r") and not rec.busy:
                print("버튼 다시 인식"); rec.start(fr.copy(), targets[min(ti, len(targets) - 1)]); phase = "RECOG2"
            if key == ord("n"):                                  # 다음 목표로 건너뛰기
                ti = min(ti + 1, len(targets) - 1); rec.start(fr.copy(), targets[ti]); phase = "RECOG2"
            if key == ord("s"):                                  # 지금 화면 원본 저장 (DB 정리용)
                d = os.path.join(args.outdir, "captures"); os.makedirs(d, exist_ok=True)
                p = os.path.join(d, f"cap_{n_cap:03d}.png"); cv2.imwrite(p, fr); n_cap += 1; print("저장:", p)
            if key in (ord("o"), ord("x")):                      # 버튼 인식 결과 맞음(o) / 틀림(x) 기록
                rname = target["name"] if target is not None else ""
                marks.append(f"{t:.2f},{targets[min(ti, len(targets) - 1)]},{rname},{'버튼맞음' if key == ord('o') else '버튼틀림'}")
                print("기록:", marks[-1])
            if key == ord("t"):                                  # 손끝 위치가 틀린 순간 기록 + 화면 저장
                d = os.path.join(args.outdir, "tip_wrong"); os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, f"{t:07.2f}.jpg"), vis); marks.append(f"{t:.2f},{targets[min(ti, len(targets) - 1)]},,손끝틀림")
                print("기록:", marks[-1])
        else:
            if writer is None:
                h, w = vis.shape[:2]; W = int(w * args.scale) // 2 * 2; Hh = int(h * args.scale) // 2 * 2
                silent = os.path.join(args.outdir, "_silent.mp4")
                writer = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, Hh))
            writer.write(cv2.resize(vis, (W, Hh)))
        n += 1
    cap.release()
    print(f"손끝을 찾은 프레임: {method_count} / 전체 {n} (끝점 보정으로 위치를 고친 프레임 {corrected})")
    open(os.path.join(args.outdir, "tips.csv"), "w", encoding="utf-8").write("\n".join(tip_log))
    if live:
        if raw_writer is not None: raw_writer.release(); print("카메라 원본:", os.path.join(args.outdir, "live_raw.mp4"))
        open(os.path.join(args.outdir, "marks.csv"), "w", encoding="utf-8-sig").write("\n".join(marks))
        ok_b = sum(1 for m in marks[1:] if m.endswith("버튼맞음")); bad_b = sum(1 for m in marks[1:] if m.endswith("버튼틀림"))
        bad_t = sum(1 for m in marks[1:] if m.endswith("손끝틀림"))
        print(f"기록 요약: 버튼 맞음 {ok_b} / 틀림 {bad_b}, 손끝 틀림 표시 {bad_t}회 → marks.csv")
        print("인식한 화면과 버튼 목록:", os.path.join(args.outdir, "screens"))
        cv2.destroyAllWindows(); return
    writer.release()
    dur = n / fps
    ev = [(et - start_t, k, x) for et, k, x in events]
    json.dump([dict(t=round(a, 2), kind=b, text=c) for a, b, c in ev], open(os.path.join(args.outdir, "events.json"), "w"), ensure_ascii=False, indent=1)
    wav = os.path.join(args.outdir, "_audio.wav"); voiced = render_audio(ev, dur, wav, cache)
    name = os.path.splitext(os.path.basename(args.video))[0]
    out = os.path.join(args.outdir, f"guide_{name}.mp4")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", out], check=True)
        print("완성:", out)
    except Exception:
        print("ffmpeg가 없어 영상과 소리를 따로 저장했습니다:", silent, wav)
    if not voiced: print("※ 한국어 음성 엔진을 찾지 못해 음성은 자막으로만 표시했습니다 (pip install pyttsx3)")
    print("안내 기록:", [f"{a:.1f}s {c or '삐'}" for a, b, c in ev if b == "say"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--camera", type=int)
    ap.add_argument("--target", help="목표 버튼 이름 예: COFFEE")
    ap.add_argument("--targets", help='목표 순서 (쉼표로 구분) 예: "주문하시려면,확인,COFFEE,아이스티노,선택완료,결제,결제하기"')
    ap.add_argument("--no-auto", action="store_true", help="화면이 바뀌어도 자동으로 다시 인식하지 않음")
    ap.add_argument("--ref", type=float, help="버튼 인식에 쓸 기준 프레임 시각(초). 없으면 손이 안 보이는 첫 프레임")
    ap.add_argument("--start", type=float, default=0); ap.add_argument("--end", type=float)
    ap.add_argument("--scale", type=float, default=0.5, help="출력 영상 크기 배율")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--hand-conf", type=float, default=0.3, help="손 인식 기준값 (낮출수록 잘 잡지만 오인식 증가)")
    ap.add_argument("--no-refine", action="store_true", help="손끝 끝점 보정 끄기 (비교용)")
    ap.add_argument("--no-mp", action="store_true", help="MediaPipe 끄기 (피부색 + 따라가기만 사용)")
    ap.add_argument("--no-skin", action="store_true", help="피부색 보완 끄기 (스켈레톤만 사용)")
    ap.add_argument("--screenrec", action="store_true", help="아이폰 화면 녹화본이면 지정 (배너·말풍선 영역 제외)")
    ap.add_argument("--width", type=int, default=1920); ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--no-record", action="store_true", help="실시간 모드에서 카메라 원본 저장 안 함")
    ap.add_argument("--list-cams", action="store_true", help="연결된 카메라 번호와 해상도 확인")
    ap.add_argument("--no-roi", action="store_true", help="키오스크 화면 영역을 찾지 않고 사진 전체에서 글자 인식 (화면이 어둡거나 영역을 잘못 잡을 때)")
    ap.add_argument("--change-thr", type=float, default=0.15, help="화면이 이 비율(0~1) 이상 바뀌면 '화면 바뀜'으로 판단 (기본 0.15)")
    ap.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "any"], help="카메라 여는 방식 (초록 화면이면 바꿔 보기)")
    ap.add_argument("--menu", default=os.path.join(HERE, "kiosk_menu.json"), help="메뉴 사전 파일 (OCR 글자 보정). 파일이 없으면 보정 안 함")
    ap.add_argument("--no-menu", action="store_true", help="메뉴 사전 보정 끄기 (비교용)")
    ap.add_argument("--server", help="BE 주소 예: http://127.0.0.1:8000  (지정하면 다음 목표를 서버가 정함)")
    ap.add_argument("--goal", help='서버 모드 주문 목표 예: "아이스 아메리카노"')
    ap.add_argument("--answers", help='서버 질문(NEED_CHOICE)에 차례로 쓸 답, 쉼표로 구분 예: "선택안함,다크나이트". 없으면 키보드로 입력')
    ap.add_argument("--tab-fallback", action="store_true", help="서버가 NOT_FOUND면 메뉴 사전으로 카테고리 탭을 AI가 임시로 안내 (BE 탭 안내 완성 전까지)")
    ap.add_argument("--planner", choices=["gpt", "rule"], help="BE 없이 노트북에서 다음 버튼을 정함. gpt=GPT에게 물어봄(OPENAI_API_KEY 필요), rule=메뉴 사전 규칙")
    ap.add_argument("--gpt-model", default="gpt-4o-mini", help="--planner gpt 에서 쓸 모델")
    ap.add_argument("--no-boost", action="store_true", help="버튼별 다시 읽기 끄기 (비교용, 인식이 조금 빨라짐)")
    ap.add_argument("--tip-shift", default="0,0", help='손끝 위치 보정(px) "dx,dy". check_tips.py eval 결과의 "추천 보정값"을 넣음')
    ap.add_argument("--click-margin", type=float, help="클릭 판정 여유 (기본 0: 손끝이 버튼 안에 들어와야 클릭). 옆 버튼을 자꾸 누르면 0.15")
    a = ap.parse_args(); a.tracker = None
    if a.list_cams:
        for i in range(6):
            c = open_camera(i, a.backend)
            if c.isOpened():
                c.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); c.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height); ok, _ = c.read()
                for _ in range(30): ok2, f2 = c.read()          # 켜지는 동안 잠깐 단색일 수 있어 몇 장 넘김
                state = "(영상 안 나옴)" if not ok else ("(단색 화면 - 초록/검정)" if blank_frame(f2 if ok2 else None) else "(영상 나옴)")
                print(f"카메라 {i}: {int(c.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))} {state}")
            c.release()
        sys.exit()
    button_v0.USE_OVERLAY_MASK = a.screenrec
    button_v0.FULL_FRAME = a.no_roi
    CHANGE_THR = a.change_thr
    if not a.video and a.camera is None: ap.error("--video 또는 --camera 필요")
    if not a.no_menu and a.menu and os.path.exists(a.menu):
        from menu_dict import MenuDict
        MENU = MenuDict(a.menu); print(f"메뉴 사전 보정 켜짐: {a.menu} (이름 {len(MENU.terms)}개)")
    elif not a.no_menu: print(f"메뉴 사전 파일이 없어 보정 없이 실행: {a.menu}")
    BOOST = not a.no_boost
    TIP_SHIFT = tuple(float(v) for v in a.tip_shift.split(","))
    if a.click_margin is not None: CFG["click_margin"] = a.click_margin
    if a.planner:
        if not a.goal: ap.error("--planner 를 쓰면 --goal 필요")
        if MENU is None: ap.error("--planner 는 메뉴 사전(kiosk_menu.json)이 필요해요")
        from planner import LocalPlanner
        PLAN = LocalPlanner(a.goal, MENU, use_gpt=(a.planner == "gpt"), model=a.gpt_model)
    elif a.server:
        if not a.goal: ap.error("--server 를 쓰면 --goal 필요")
        from be_client import BEPlanner
        PLAN = BEPlanner(a.server, a.goal, answers=[x.strip() for x in (a.answers or "").split(",") if x.strip()],
                         menu=MENU, tab_fallback=a.tab_fallback)
    elif not a.target and not a.targets: ap.error("--target 또는 --targets 필요 (계획 모드면 --planner, --goal / 서버 모드면 --server, --goal)")
    run(a)

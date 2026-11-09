"""손끝 인식 방식 3개 비교 (한 화면에 같이 표시)
  빨강   : 기준 화면 비교 (tip_check.py 방식)
  파랑   : MediaPipe
  보라   : SCUT-Ego-Gesture로 학습된 공개 모델 그대로 (Unified Gesture and Fingertip Detection, 손 상자 + 손끝)
  초록   : 주황 + B(손가락 방향으로 손가락 색이 끝나는 곳까지 밀기). 주황은 그대로 두고 따로 기록 → tips_ugfd3.csv
  주황   : 공개 모델 개선판 = ① 검은 여백 잘라내기 ② 정사각형 손 상자 + 손끝 쪽으로 한 번 더 ③ 손끝 경계 보정
           ④ 모델 결과 사이 프레임은 따라가기 ⑤ 직전 손 주변에서 먼저 찾기

실행 (test6 폴더에서)
  python compare_tips.py --camera 1 --ugfd C:\\songil\\Unified-Gesture-and-Fingertip-Detection
  python compare_tips.py --video out\\cmp1\\live_raw.mp4 --ugfd ...      (녹화본: 한 장씩 전부 계산, 느려도 정확한 비교)
  python compare_tips.py --video out\\cmp\\live_raw.mp4 --outdir out\\vis --save-vis --labels labels_5-25.json --no-green
        → 점·상자·정답(흰 십자)이 그려진 영상 out\\vis\\vis.mp4 저장 (팀 공유·발표용)

키
  o : 보라(원본) 켜기/끄기 (끄면 주황이 더 자주 갱신됨)
  p : 주황 ② 한 번 더 찾기 켜기/끄기     e : 주황 ③ 경계 보정 켜기/끄기     f : 주황 ④ 따라가기 켜기/끄기
  r : 공개 모델에 넣는 방향 바꾸기 (0도 → 90도 → 180도 → 270도). 학습 사진은 손가락이 위를 향해서, 방향에 따라 결과가 달라질 수 있음
  [ ] : 공개 모델 손 찾기 기준 낮추기/높이기 (기본 0.5, 원래 코드는 0.8)
  b : 기준 화면 다시 찍기 (빨강용)   t : 틀린 장면 저장   q : 종료

종료하면 out\\cmp\\ 에 live_raw.mp4, tips.csv(빨강), tips_mp.csv(파랑), tips_ugfd.csv(보라), tips_ugfd2.csv(주황) 저장
→ python check_tips.py label 로 정답을 찍고 eval 을 세 파일에 각각 돌리면 오차를 숫자로 비교할 수 있어요
"""
import argparse, os, sys, time, threading, functools, math, json
import cv2, numpy as np
from guide_sim import open_camera, OneEuro, put_kr, HandTracker, HERE, refine_tip, TipFlow, unit
from tip_check import DiffTip, RotatedCap

ROTS = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]


class UGFD:
    """공개 모델: YOLO로 손 상자 → 상자 안에서 손가락 5개 끝 위치와 '펴졌는지' 확률. 검지(두 번째) 끝만 사용"""
    def __init__(self, root, hand_thr=0.5):
        if not os.path.exists(os.path.join(root, "weights", "fingertip.h5")):
            sys.exit(f"가중치가 없어요: {os.path.join(root, 'weights', 'fingertip.h5')}  (weights 폴더 안에 fingertip.h5, yolo.h5)")
        sys.path.insert(0, root)
        import net.network as nn
        nn.VGG16 = functools.partial(nn.VGG16, weights=None)      # 기본 VGG16 가중치는 어차피 덮어쓰므로 받지 않음
        from unified_detector import Fingertips
        from hand_detector.detector import YOLO
        self.hand = YOLO(weights=os.path.join(root, "weights", "yolo.h5"), threshold=hand_thr)
        self.fing = Fingertips(weights=os.path.join(root, "weights", "fingertip.h5"))
        for m in (self.hand.model, self.fing.model):              # model.predict 는 느려서 직접 호출로 바꿈
            m.predict = (lambda mm: (lambda x, **k: (lambda o: [t.numpy() for t in o] if isinstance(o, (list, tuple)) else o.numpy())(mm(x, training=False))))(m)
        self.rot = 0; self.box = None; self.all_tips = []

    def find(self, bgr):
        """반환: 검지 끝 (x, y) 원본 좌표 또는 None"""
        H, W = bgr.shape[:2]; code = ROTS[self.rot]
        img = cv2.rotate(bgr, code) if code is not None else bgr
        try:
            tl, br = self.hand.detect(image=img)
        except Exception:
            tl, br = None, None
        self.box = None; self.all_tips = []
        if tl is None or br is None: return None
        x0, y0 = max(0, tl[0]), max(0, tl[1]); x1, y1 = min(img.shape[1], br[0]), min(img.shape[0], br[1])
        if x1 - x0 < 10 or y1 - y0 < 10: return None
        crop = img[y0:y1, x0:x1]
        prob, pos = self.fing.classify(image=crop)
        pos = np.mean(pos, 0); h, w = crop.shape[:2]
        pts = [(pos[2 * i] * w + x0, pos[2 * i + 1] * h + y0) for i in range(5)]
        back = lambda p: self._unrotate(p, W, H)
        self.box = [back((x0, y0)), back((x1, y1))]
        self.all_tips = [back(p) for i, p in enumerate(pts) if prob[i] >= 0.5]
        return back(pts[1]) if prob[1] >= 0.5 else None

    # ---------------- 개선판 (주황) ----------------
    @staticmethod
    def content_box(bgr):
        """검은 여백(세로 영상을 가로 화면에 넣을 때 생기는 양옆 검은 띠)을 뺀 실제 영상 영역"""
        g = cv2.cvtColor(cv2.resize(bgr, None, fx=0.25, fy=0.25), cv2.COLOR_BGR2GRAY) > 16
        cols = np.nonzero(g.mean(0) > 0.05)[0]; rows = np.nonzero(g.mean(1) > 0.05)[0]
        if len(cols) < 10 or len(rows) < 10: return (0, 0, bgr.shape[1], bgr.shape[0])
        return (int(cols[0] * 4), int(rows[0] * 4), int(min(bgr.shape[1], (cols[-1] + 1) * 4)), int(min(bgr.shape[0], (rows[-1] + 1) * 4)))

    def _yolo(self, img, x0, y0, x1, y1):
        if x1 - x0 < 32 or y1 - y0 < 32: return None
        try: tl, br = self.hand.detect(image=img[y0:y1, x0:x1])
        except Exception: return None
        if tl is None or br is None: return None
        return (tl[0] + x0, tl[1] + y0, br[0] + x0, br[1] + y0)

    def _tips_square(self, img, cx, cy, side):
        """(cx,cy) 중심 정사각형을 잘라(밖은 가장자리 색으로 채움) 손끝 모델에 넣음 → (확률 5개, 손끝 5개 좌표)"""
        s = int(max(32, side)); x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2)); H, W = img.shape[:2]
        pl, pt, pr, pb = max(0, -x0), max(0, -y0), max(0, x0 + s - W), max(0, y0 + s - H)
        crop = img[max(0, y0):min(H, y0 + s), max(0, x0):min(W, x0 + s)]
        if pl or pt or pr or pb: crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
        prob, pos = self.fing.classify(image=crop); pos = np.mean(pos, 0)
        return prob, [(pos[2 * i] * s + x0, pos[2 * i + 1] * s + y0) for i in range(5)]

    TIP_THR = 0.3   # 검지 '펴짐' 확률 기준 (원래 코드 0.5)

    def _tip_in_box(self, img, box, twice):
        """손 상자 하나로 검지 끝 찾기: ② 정사각형(+손끝 쪽 한 번 더) → 안 되면 원본 방식(상자 그대로)"""
        bx0, by0, bx1, by1 = box; bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2; side = 1.1 * max(bx1 - bx0, by1 - by0)
        prob, pts = self._tips_square(img, bcx, bcy, side)
        if prob[1] >= self.TIP_THR:
            tip = pts[1]
            if twice:
                c2 = (bcx + 0.3 * (tip[0] - bcx), bcy + 0.3 * (tip[1] - bcy))
                prob2, pts2 = self._tips_square(img, c2[0], c2[1], 0.85 * side)
                if prob2[1] >= self.TIP_THR and math.dist(pts2[1], tip) < 0.15 * side: tip = ((tip[0] + pts2[1][0]) / 2, (tip[1] + pts2[1][1]) / 2)
            return tip, "정사각형"
        x0, y0, x1, y1 = [int(v) for v in box]                     # ④ 원본 방식: 상자 그대로 잘라서
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
        if x1 - x0 < 10 or y1 - y0 < 10: return None, ""
        prob, pos = self.fing.classify(image=img[y0:y1, x0:x1]); pos = np.mean(pos, 0)
        if prob[1] >= self.TIP_THR: return (pos[2] * (x1 - x0) + x0, pos[3] * (y1 - y0) + y0), "원본방식"
        return None, ""

    def find2(self, bgr, prev_box=None, twice=True):
        """반환: (검지 끝, 손 상자(x0,y0,x1,y1), 손 안쪽 점) - 모두 원본 좌표. 못 찾으면 (None, None, None)
        순서: 직전 손 주변 → (실패하면) 검은 여백 뺀 전체 화면. 상자마다 정사각형 → 원본 방식"""
        H, W = bgr.shape[:2]; code = ROTS[self.rot]
        img = cv2.rotate(bgr, code) if code is not None else bgr
        cx0, cy0, cx1, cy1 = self.content_box(img)
        tries = []
        if prev_box is not None:                                     # ⑤ 직전 손 주변(손 크기의 2.2배)
            pb = [self._rot_pt(p, W, H) for p in ((prev_box[0], prev_box[1]), (prev_box[2], prev_box[3]))]
            bx0, by0 = np.min(pb, 0); bx1, by1 = np.max(pb, 0)
            side = max(300, 2.2 * max(bx1 - bx0, by1 - by0)); mx, my = (bx0 + bx1) / 2, (by0 + by1) / 2
            tries.append((int(max(cx0, mx - side / 2)), int(max(cy0, my - side / 2)), int(min(cx1, mx + side / 2)), int(min(cy1, my + side / 2))))
        tries.append((cx0, cy0, cx1, cy1))                           # ③ 전체 화면(검은 여백 제외)
        last_box = None; self.how = ""
        for region in tries:
            box = self._yolo(img, *region)
            if box is None: continue
            last_box = box
            tip, how = self._tip_in_box(img, box, twice)
            if tip is not None:
                self.how = how
                bcx, bcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                return self._unrotate(tip, W, H), self._back_box(box, W, H), self._unrotate((bcx, bcy), W, H)
        return None, (self._back_box(last_box, W, H) if last_box else None), None

    def _back_box(self, box, W, H):
        a = self._unrotate((box[0], box[1]), W, H); b = self._unrotate((box[2], box[3]), W, H)
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))

    def _rot_pt(self, p, W, H):
        """원본 좌표 → 돌린 그림 좌표"""
        x, y = p
        if self.rot == 0: return (x, y)
        if self.rot == 1: return (H - 1 - y, x)
        if self.rot == 2: return (W - 1 - x, H - 1 - y)
        return (y, W - 1 - x)

    def _unrotate(self, p, W, H):
        x, y = p
        if self.rot == 0: return (x, y)
        if self.rot == 1: return (y, H - 1 - x)          # 시계 90 돌린 그림 좌표 → 원본
        if self.rot == 2: return (W - 1 - x, H - 1 - y)
        return (W - 1 - y, x)                            # 반시계 90


def extend_tip(bgr, tip, d, back=(8, 18), max_ext=70, thr=18.0, half=3):
    """B: 손끝이 손가락 안쪽에 찍혔을 때, 손가락 방향(d)으로 '손가락 색이 끝나는 곳'까지 밀어 줌.
    손끝 조금 안쪽(back px)의 색을 손가락 색으로 잡고, 앞으로 한 칸씩 가며 그 색과 달라지는 첫 지점을 찾음.
    살색 기준을 쓰지 않아서 파란 화면 빛에 물든 손가락에도 동작. 반환: (새 손끝, 민 거리) 또는 (None, 0)"""
    H, W = bgr.shape[:2]; d = np.float32(d); nrm = np.float32([-d[1], d[0]]); tip = np.float32(tip)
    r = back[1] + max_ext + 10
    x0, y0 = int(max(0, tip[0] - r)), int(max(0, tip[1] - r)); x1, y1 = int(min(W, tip[0] + r)), int(min(H, tip[1] + r))
    if x1 - x0 < 20 or y1 - y0 < 20: return None, 0
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr[y0:y1, x0:x1], (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]; o = np.float32([x0, y0])
    def sample(p):
        vals = []
        for k in range(-half, half + 1):
            q = p + nrm * k - o; xi, yi = int(round(q[0])), int(round(q[1]))
            if 0 <= xi < w and 0 <= yi < h: vals.append(lab[yi, xi])
        return np.mean(vals, 0) if vals else None
    refs = [sample(tip - d * b) for b in range(back[0], back[1] + 1, 2)]
    refs = [v for v in refs if v is not None]
    if len(refs) < 3: return None, 0
    ref = np.median(refs, 0)
    if np.linalg.norm(np.std(refs, 0)) > thr: return None, 0          # 안쪽 색이 고르지 않음 = 손가락 위가 아님
    end, miss = 0, 0
    for s_ in range(-4, max_ext + 1):
        v = sample(tip + d * s_)
        if v is None: break
        if np.linalg.norm(v - ref) <= thr: end, miss = s_, 0
        else:
            miss += 1
            if miss >= 3: break
    if end <= 2: return None, 0                                        # 이미 끝에 있음
    if end >= max_ext: return None, 0                                  # 끝을 못 찾음 (배경이 손가락 색과 비슷)
    return (float(tip[0] + d[0] * end), float(tip[1] + d[1] * end)), end


class Improved:
    """주황: 모델 결과(느림) + 그 사이 프레임 따라가기(빠름) + 경계 보정 + 잠깐 놓치면 마지막 위치 유지"""
    HOLD = 6    # 놓쳤을 때 마지막 위치를 유지할 최대 프레임 수 (30fps 기준 0.2초)
    SLOW = 4.0  # 직전 손끝 속도가 이 값(px/프레임, 30fps면 초당 120px) 이하일 때만 유지. 움직이던 중이면 유지하지 않음

    def __init__(self, ug):
        self.ug = ug; self.flow = TipFlow(max_frames=40); self.buf = {}; self.box = None; self.toward = None
        self.twice = True; self.edge = True; self.follow = True; self.cur = None; self.model_tip = None
        self.last = None; self.last_n = -999; self.held = False; self.speed = 0.0
        self.ext = None; self.ext_len = 0     # B 결과 (주황을 손가락 끝 쪽으로 민 것, 초록)

    def push_gray(self, n, gray):
        self.buf[n] = gray
        for k in [k for k in self.buf if k < n - 60]: del self.buf[k]

    def apply_result(self, n0, n, res):
        """n0번 프레임의 모델 결과를 받아, n0 → 지금(n)까지 따라가서 지금 위치로 맞춤"""
        tip, box, toward = res; self.model_tip = tip
        if box is not None or tip is not None: self.box = box
        if tip is None: return                      # ① 모델이 실패해도 따라가던 위치는 그대로
        self.toward = toward
        if not self.follow or n0 not in self.buf: self.cur = tip; self.flow.stop(); return
        self.flow.start(self.buf[n0], tip, toward); cur = tip
        for k in range(n0 + 1, n + 1):
            if k not in self.buf: break
            c = self.flow.track(self.buf[k])
            if c is None: break
            cur = c
        self.cur = cur

    def step(self, n, frame):
        """새 프레임마다: 따라가기로 위치 갱신 + 경계 보정. 반환: 표시할 손끝"""
        if self.follow and self.flow.pts is not None and self.flow.prev is not self.buf.get(n):
            c = self.flow.track(self.buf[n]); self.cur = c
        elif self.follow and self.flow.pts is None and self.cur is not None and self.last_n < n - 1:
            self.cur = None                         # 따라가기 끝난 뒤 새 결과 없음
        self.held = False; self.ext = None; self.ext_len = 0
        if self.cur is None:
            # ② 잠깐 놓치면 마지막 위치 유지. 단, 손이 거의 멈춰 있었을 때만 (움직이던 중이면 예전 위치가 틀린 정보라 버림)
            if self.last is not None and n - self.last_n <= self.HOLD and self.speed <= self.SLOW:
                self.held = True; self.ext = self.last; return self.last
            return None
        tip = self.cur
        if self.edge and self.toward is not None:
            d = unit(np.subtract(self.cur, self.toward))
            if d is not None:
                fx = refine_tip(frame, self.cur, d, win_ratio=0.04)
                if fx is not None and math.dist(fx, self.cur) < 40: tip = fx
        if self.toward is not None:                                   # B: 초록 = 주황을 손가락 끝까지 밀어 본 것 (비교용, 주황은 그대로)
            d = unit(np.subtract(tip, self.toward))
            if d is not None:
                e, L = extend_tip(frame, tip, d)
                self.ext, self.ext_len = (e, L) if e is not None else (tip, 0)
        if self.last is not None and n - self.last_n <= self.HOLD:
            v = math.dist(tip, self.last) / max(1, n - self.last_n); self.speed = 0.5 * self.speed + 0.5 * v
        else: self.speed = 0.0
        self.last, self.last_n = tip, n
        return tip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int); ap.add_argument("--video")
    ap.add_argument("--ugfd", default=r"C:\songil\Unified-Gesture-and-Fingertip-Detection", help="공개 모델 폴더 (weights 폴더가 들어 있는 곳)")
    ap.add_argument("--rotate", choices=["cw", "ccw", "180"])
    ap.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "any"])
    ap.add_argument("--width", type=int, default=1920); ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--hand-thr", type=float, default=0.5, help="공개 모델 손 찾기 기준 (원래 코드 0.8)")
    ap.add_argument("--every", type=int, default=7, help="녹화본 모드: 모델을 몇 장마다 돌릴지 (실시간 250ms ≈ 7장과 같게)")
    ap.add_argument("--outdir", default=os.path.join("out", "cmp"))
    ap.add_argument("--save-vis", action="store_true", help="점·상자가 그려진 화면을 영상으로 저장 (vis.mp4)")
    ap.add_argument("--labels", help="정답 파일(labels_*.json)을 주면 정답 위치를 흰 십자로 같이 표시")
    ap.add_argument("--no-green", action="store_true", help="초록(B) 표시 안 함")
    ap.add_argument("--no-window", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.camera is None and not a.video: ap.error("--camera 또는 --video 필요")
    os.makedirs(os.path.join(a.outdir, "tip_wrong"), exist_ok=True)
    live = a.camera is not None
    print("모델 불러오는 중...")
    ug = UGFD(a.ugfd, a.hand_thr); mp = HandTracker(os.path.join(HERE, "hand_landmarker.task")); det = DiffTip(); imp = Improved(ug)
    cap = open_camera(a.camera, a.backend) if live else cv2.VideoCapture(a.video)
    if not cap.isOpened(): sys.exit("영상을 열 수 없습니다")
    if live: cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if a.rotate: cap = RotatedCap(cap, a.rotate)
    fps_out = 30.0 if live else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    opts = {"orig": True}

    def run_models(f):
        """한 프레임에 모델 돌리기: (원본 결과, 개선판 결과, 걸린 ms)"""
        t0 = time.time()
        o = (ug.find(f), ug.box, list(ug.all_tips)) if opts["orig"] else (None, None, [])
        r = ug.find2(f, imp.box, imp.twice)
        return o, r, (time.time() - t0) * 1000

    # 실시간: 모델은 따로 돌림 (가장 최근 프레임만). 녹화본: --every 장마다 그 자리에서 돌림 (실시간과 같은 조건)
    shared = {"job": None, "out": None, "stop": False, "ms": 0.0}
    def worker():
        while not shared["stop"]:
            job = shared["job"]
            if job is None: time.sleep(0.005); continue
            shared["job"] = None; n0, f = job
            o, r, ms = run_models(f); shared["out"] = (n0, o, r); shared["ms"] = ms
    if live: threading.Thread(target=worker, daemon=True).start()

    writer = None; logs = {k: ["t,x,y,method"] for k in ("diff", "mp", "ugfd", "ugfd2", "ugfd3")}
    n = 0; ref_buf = []; t_last = time.time(); fps_view = 0.0; n_wrong = 0; cnt = dict.fromkeys(logs, 0)
    orig = (None, None, []); vis_writer = None
    gt_labels = {float(k): v for k, v in json.load(open(a.labels)).items()} if a.labels else {}
    print("손을 화면에서 뺀 상태로 1~2초 기다리면 기준 화면을 찍어요")
    while True:
        ok, fr = cap.read()
        if not ok: break
        t = n / fps_out
        if live:
            if writer is None: writer = cv2.VideoWriter(os.path.join(a.outdir, "live_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (fr.shape[1], fr.shape[0]))
            writer.write(fr)
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); imp.push_gray(n, gray)
        if det.ref is None:
            g = cv2.resize(gray, (160, 160))
            if g.std() >= 5 and ((not live) or t >= 1.0): ref_buf.append(fr.copy())
            if len(ref_buf) >= 15: det.set_ref(ref_buf); ref_buf = []; print("기준 화면 찍음")
        d_tip = det.find(fr)
        h = mp.detect(fr); m_tip = h[8] if h else None
        # 공개 모델
        if live:
            if shared["job"] is None: shared["job"] = (n, fr.copy())
            out = shared["out"]
            if out is not None:
                shared["out"] = None; n0, orig, r = out; imp.apply_result(n0, n, r)
        elif n % max(1, a.every) == 0:
            orig, r, shared["ms"] = run_models(fr); imp.apply_result(n, n, r)
        i_tip = imp.step(n, fr)
        u_tip, u_box, u_all = orig
        e_tip = imp.ext if i_tip is not None else None
        for k, p, name in (("diff", d_tip, "차이"), ("mp", m_tip, "스켈레톤"), ("ugfd", u_tip, "공개모델"), ("ugfd2", i_tip, "개선"), ("ugfd3", e_tip, "개선+B")):
            logs[k].append(f"{t:.3f},{'' if p is None else round(p[0], 1)},{'' if p is None else round(p[1], 1)},{'' if p is None else name}")
            if p is not None: cnt[k] += 1
        n += 1
        if a.no_window and not a.save_vis: continue
        vis = fr.copy()
        if u_box: cv2.rectangle(vis, tuple(map(int, np.min(u_box, 0))), tuple(map(int, np.max(u_box, 0))), (200, 0, 200), 1)
        if imp.box: cv2.rectangle(vis, (int(imp.box[0]), int(imp.box[1])), (int(imp.box[2]), int(imp.box[3])), (0, 140, 255), 2)
        if u_tip is not None: cv2.circle(vis, (int(u_tip[0]), int(u_tip[1])), 11, (200, 0, 200), -1)
        if m_tip is not None: cv2.circle(vis, (int(m_tip[0]), int(m_tip[1])), 9, (255, 80, 0), -1)
        if d_tip is not None: cv2.circle(vis, (int(d_tip[0]), int(d_tip[1])), 7, (0, 0, 255), -1)
        if i_tip is not None:
            cv2.circle(vis, (int(i_tip[0]), int(i_tip[1])), 9, (0, 140, 255), -1 if not imp.held else 2)   # 속 빈 원 = 잠깐 놓쳐서 마지막 위치 유지 중
            cv2.circle(vis, (int(i_tip[0]), int(i_tip[1])), 11, (255, 255, 255), 2)
        if e_tip is not None and imp.ext_len > 0 and not a.no_green:                       # 초록: B로 민 위치 + 민 선
            cv2.line(vis, (int(i_tip[0]), int(i_tip[1])), (int(e_tip[0]), int(e_tip[1])), (0, 220, 0), 2)
            cv2.circle(vis, (int(e_tip[0]), int(e_tip[1])), 7, (0, 220, 0), -1)
        now = time.time(); fps_view = 0.9 * fps_view + 0.1 / max(1e-3, now - t_last); t_last = now
        ok3 = lambda p: "O" if p is not None else "-"
        onoff = lambda v: "켬" if v else "끔"
        line = (("" if a.no_green else f"초록(B, +{imp.ext_len}px)  ") + f"주황(개선) {ok3(i_tip)}  보라(원본) {ok3(u_tip) if opts['orig'] else '끔'}  파랑 {ok3(m_tip)}  빨강 {ok3(d_tip)}"
                f" | 모델 {shared['ms']:.0f}ms ({getattr(ug, 'how', '')}) | {fps_view:.0f} fps")
        line2 = f"o 보라 {onoff(opts['orig'])}  p 한번더 {onoff(imp.twice)}  e 경계보정 {onoff(imp.edge)}  f 따라가기 {onoff(imp.follow)}  r 방향 {ug.rot * 90}도  [ ] 기준 {ug.hand.threshold:.2f}  t 틀림  q 종료"
        vis = put_kr(vis, line, (10, 10), 22); vis = put_kr(vis, line2, (10, 42), 18)
        if gt_labels:                                                 # 정답(흰 십자): 가장 가까운 정답 시각이 0.13초 이내일 때
            kt = min(gt_labels, key=lambda k: abs(k - t))
            if abs(kt - t) <= 0.13 and gt_labels[kt] is not None:
                gx, gy = int(gt_labels[kt][0]), int(gt_labels[kt][1])
                cv2.drawMarker(vis, (gx, gy), (255, 255, 255), cv2.MARKER_CROSS, 34, 3); cv2.drawMarker(vis, (gx, gy), (0, 0, 0), cv2.MARKER_CROSS, 34, 1)
        if a.save_vis:
            if vis_writer is None: vis_writer = cv2.VideoWriter(os.path.join(a.outdir, "vis.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (vis.shape[1], vis.shape[0]))
            vis_writer.write(vis)
        if a.no_window: continue
        sc = min(1.0, 1400 / vis.shape[1], 900 / vis.shape[0])
        cv2.imshow("songil tip compare", cv2.resize(vis, None, fx=sc, fy=sc))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        if k == ord("o"): opts["orig"] = not opts["orig"]; orig = (None, None, [])
        if k == ord("p"): imp.twice = not imp.twice
        if k == ord("e"): imp.edge = not imp.edge
        if k == ord("f"): imp.follow = not imp.follow
        if k == ord("r"): ug.rot = (ug.rot + 1) % 4; imp.box = None; print("공개 모델 방향:", ug.rot * 90, "도")
        if k == ord("["): ug.hand.threshold = max(0.05, ug.hand.threshold - 0.05)
        if k == ord("]"): ug.hand.threshold = min(0.95, ug.hand.threshold + 0.05)
        if k == ord("b"): det.ref = None; ref_buf = []
        if k == ord("t"): cv2.imwrite(os.path.join(a.outdir, "tip_wrong", f"wrong_{n_wrong:03d}_{t:.1f}s.jpg"), vis); n_wrong += 1
    shared["stop"] = True
    if writer: writer.release()
    if vis_writer: vis_writer.release(); print("표시 영상 저장:", os.path.join(a.outdir, "vis.mp4"))
    for k, fn in (("diff", "tips.csv"), ("mp", "tips_mp.csv"), ("ugfd", "tips_ugfd.csv"), ("ugfd2", "tips_ugfd2.csv"), ("ugfd3", "tips_ugfd3.csv")):
        open(os.path.join(a.outdir, fn), "w", encoding="utf-8").write("\n".join(logs[k]))
    print(f"손끝 찾은 프레임 (전체 {n}장): 주황(개선) {cnt['ugfd2']}, 보라(원본) {cnt['ugfd']}, 파랑(MediaPipe) {cnt['mp']}, 빨강(차이) {cnt['diff']}")
    print("저장:", a.outdir, "(정답 찍기: python check_tips.py label --video", os.path.join(a.outdir, "live_raw.mp4"), "...)")
    if not a.no_window: cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""손끝 인식만 확인하는 코드 (버튼 인식·안내 없음)

방식: "손이 없을 때 화면(기준)"과 지금 화면을 비교 → 달라진 부분 = 손
      → 손이 들어온 화면 가장자리에서 가장 멀리 뻗은 점 = 손끝 (빨간 점)
      관절을 찾지 않아서 손바닥이 안 보여도 손가락만 들어오면 잡힘
비교용: --mp 를 붙이면 MediaPipe 손끝(파란 점)도 같이 표시

실행
  python tip_check.py --camera 1 --rotate cw
  python tip_check.py --camera 1 --rotate cw --mp          (MediaPipe와 비교)
  python tip_check.py --video out\\tip1\\live_raw.mp4       (녹화본 다시 보기)

키 (영상 창을 클릭한 상태에서)
  b : 기준 화면 다시 찍기 (손을 화면에서 뺀 상태에서. 카메라가 움직였거나 화면이 바뀌면)
  t : 지금 손끝이 틀림 → 장면 저장 (tip_wrong/)
  [ ] : 색 차이 기준 낮추기/높이기 (손이 덜 잡히면 낮추고, 손 아닌 곳이 잡히면 높이기)
  m : 손 영역(초록) 표시 켜기/끄기
  q : 종료 (tips.csv 저장 → check_tips.py eval 로 오차 계산 가능)
"""
import argparse, os, sys, time, math
import cv2, numpy as np
from guide_sim import open_camera, OneEuro, put_kr, HandTracker, HERE


class RotatedCap:
    """카메라/영상을 돌려서 읽기 (--rotate). guide_sim 버전과 상관없이 쓰려고 여기에도 둠"""
    CODES = {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}
    def __init__(self, cap, how): self.cap, self.code = cap, self.CODES[how]
    def read(self):
        ok, f = self.cap.read()
        return ok, (cv2.rotate(f, self.code) if ok and f is not None else f)
    def get(self, p): return self.cap.get(p)
    def set(self, *a): return self.cap.set(*a)
    def isOpened(self): return self.cap.isOpened()
    def release(self): self.cap.release()


class DiffTip:
    """기준 화면과의 차이로 손 영역을 찾고, 들어온 쪽에서 가장 먼 점을 손끝으로"""
    def __init__(self, thr_ab=10.0, thr_bright=35.0, thr_dark=60.0, side="auto"):
        self.ref = None; self.thr_ab, self.thr_bright, self.thr_dark = thr_ab, thr_bright, thr_dark
        self.side = side; self.mask = None; self.entry = None; self.area = 0
        self.bad = False; self.shift = (0.0, 0.0); self.noise = None; self.ref_gray = None

    PROC = 640   # 처리 크기(긴 변). 빠르게 하려고 줄여서 계산하고 결과만 원래 크기로 되돌림
    BAD_FRAC = 0.35   # 화면의 이 비율 이상이 '달라짐'이면 손이 아니라 기준 화면이 안 맞는 것 (카메라 움직임·화면 바뀜)

    def _lab(self, bgr):
        sc = min(1.0, self.PROC / max(bgr.shape[:2])); self.sc = sc
        small = cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else bgr
        return cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)

    def set_ref(self, frames):
        labs = [self._lab(f) for f in frames]; self.ref_gray = labs[0][..., 0].copy()
        st = np.stack([labs[0]] + [self._align(l) for l in labs[1:]])   # 손으로 들고 찍어 흔들려도 첫 장에 맞춤
        self.ref = np.median(st, axis=0)
        # 화면 깜빡임·카메라 잡음이 큰 곳은 기준을 높임 (기준 프레임들끼리 흔들린 정도)
        self.noise = np.hypot(st[..., 1].std(0), st[..., 2].std(0))
        self.ref_gray = self.ref[..., 0].copy()

    def _align(self, lab):
        """카메라가 조금 흔들린 만큼(평행 이동) 맞춰 줌"""
        win = cv2.createHanningWindow(self.ref_gray.shape[::-1], cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(self.ref_gray, lab[..., 0].copy(), win)
        if resp < 0.05 or abs(dx) > 40 or abs(dy) > 40: self.shift = (0.0, 0.0); return lab
        self.shift = (dx, dy)
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        return cv2.warpAffine(lab, M, lab.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)

    def find(self, bgr):
        if self.ref is None: return None
        lab = self._align(self._lab(bgr)); d = lab - self.ref
        for c in range(3): d[..., c] -= np.median(d[..., c])        # 카메라 밝기·색 자동 조정 보정 (손은 화면 일부라 중앙값은 배경 기준)
        dL = d[..., 0]; dab = np.hypot(d[..., 1], d[..., 2])
        # 손: 색이 달라졌거나(피부색 vs 화면), 많이 밝아졌거나, 아주 많이 어두워짐(역광 손).
        # 조금 어두워지기만 한 곳은 그림자로 보고 뺌
        thr = np.maximum(self.thr_ab, 3 * self.noise)
        m = (dab > thr) | (dL > self.thr_bright) | (dL < -self.thr_dark)
        self.bad = m.mean() > self.BAD_FRAC
        if self.bad: self.mask = None; self.entry = None; self.area = 0; return None
        m = m.astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        H, W = m.shape
        n, lab_img, stats, _ = cv2.connectedComponentsWithStats(m)
        min_area = 0.002 * H * W
        best, best_key = None, None
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < min_area: continue
            touches = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
            key = (touches, a)                      # 화면 가장자리에서 들어온 덩어리 우선, 그다음 큰 것
            if best_key is None or key > best_key: best, best_key = i, key
        self.mask = None; self.entry = None; self.area = 0
        if best is None: return None
        comp = (lab_img == best).astype(np.uint8)
        full = (bgr.shape[1], bgr.shape[0]); k = 1.0 / self.sc
        self.mask = cv2.resize(comp, full, interpolation=cv2.INTER_NEAREST) if self.sc < 1 else comp
        self.area = int(stats[best][4] * k * k)
        # 들어온 쪽(가장자리와 닿은 부분의 가운데)
        e = self._entry(comp)
        if e is None: return None
        self.entry = (e[0] * k, e[1] * k)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = max(cnts, key=len).reshape(-1, 2).astype(np.float32)
        dist = np.hypot(pts[:, 0] - e[0], pts[:, 1] - e[1])
        far = float(dist.max())
        near = pts[dist >= far - 4]                 # 가장 먼 점 근처 몇 개 평균 (튐 방지)
        tip = near.mean(0) + np.array(self.shift, np.float32)   # 흔들림 맞춘 만큼 되돌림
        if far < 30 * self.sc: return None          # 가장자리에 살짝 걸친 정도는 손으로 안 봄
        return float(tip[0] * k), float(tip[1] * k)

    def _entry(self, comp):
        H, W = comp.shape
        sides = {"bottom": np.nonzero(comp[H - 3])[0], "top": np.nonzero(comp[2])[0],
                 "left": np.nonzero(comp[:, 2])[0], "right": np.nonzero(comp[:, W - 3])[0]}
        if self.side != "auto": s = self.side
        else: s = max(sides, key=lambda k: len(sides[k]))
        v = sides[s]
        if len(v) == 0:                             # 가장자리에 안 닿음: 덩어리에서 가장자리에 제일 가까운 점
            ys, xs = np.nonzero(comp)
            dd = np.minimum.reduce([ys, H - 1 - ys, xs, W - 1 - xs]); i = int(np.argmin(dd))
            return float(xs[i]), float(ys[i])
        c = float(np.median(v))
        return {"bottom": (c, H - 1.0), "top": (c, 0.0), "left": (0.0, c), "right": (W - 1.0, c)}[s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int); ap.add_argument("--video")
    ap.add_argument("--rotate", choices=["cw", "ccw", "180"])
    ap.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "any"])
    ap.add_argument("--width", type=int, default=1920); ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--mp", action="store_true", help="MediaPipe 손끝도 같이 표시 (비교용, 파란 점)")
    ap.add_argument("--side", default="auto", choices=["auto", "bottom", "top", "left", "right"], help="손이 들어오는 쪽 (기본: 자동)")
    ap.add_argument("--thr", type=float, default=10.0, help="색 차이 기준 (기본 10)")
    ap.add_argument("--outdir", default=os.path.join("out", "tip"))
    ap.add_argument("--no-window", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--max-frames", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.camera is None and not a.video: ap.error("--camera 또는 --video 필요")
    os.makedirs(os.path.join(a.outdir, "tip_wrong"), exist_ok=True)
    live = a.camera is not None
    cap = open_camera(a.camera, a.backend) if live else cv2.VideoCapture(a.video)
    if not cap.isOpened(): sys.exit("영상을 열 수 없습니다")
    if live:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if a.rotate: cap = RotatedCap(cap, a.rotate)
    fps_out = 30.0 if live else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    det = DiffTip(thr_ab=a.thr, side=a.side); filt = OneEuro()
    mp = HandTracker(os.path.join(HERE, "hand_landmarker.task")) if a.mp else None
    writer = None; log = ["t,x,y,method"]; log_mp = ["t,x,y,method"]
    show_mask = True; n = 0; ref_buf = []; ref_prev = None; t_last = time.time(); fps_view = 0.0; n_wrong = 0
    print("손을 화면에서 뺀 상태로 1~2초 기다리면 기준 화면을 찍어요 (다시 찍기: b)")
    while True:
        ok, fr = cap.read()
        if not ok: break
        t = n / fps_out; n += 1
        if live:
            if writer is None:
                writer = cv2.VideoWriter(os.path.join(a.outdir, "live_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (fr.shape[1], fr.shape[0]))
            writer.write(fr)
        # 기준 화면: 처음 (실시간은 1초 뒤부터) 10장의 중앙값
        if det.ref is None:
            g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (160, 160)).astype(np.float32)
            if g.std() < 5: ref_buf = []                                   # 카메라 켜지는 중 (단색 화면)
            elif (not live) or t >= 1.0:
                ref_buf.append(fr.copy())
            ref_prev = g
            if len(ref_buf) >= 15: det.set_ref(ref_buf); ref_buf = []; print("기준 화면 찍음")
        tip = det.find(fr)
        stip = filt(tip, t) if tip is not None else None
        if tip is None: filt.reset()
        log.append(f"{t:.3f},{'' if stip is None else round(stip[0], 1)},{'' if stip is None else round(stip[1], 1)},{'' if stip is None else '차이'}")
        mtip = None
        if mp is not None:
            h = mp.detect(fr); mtip = h[8] if h else None
            log_mp.append(f"{t:.3f},{'' if mtip is None else round(mtip[0], 1)},{'' if mtip is None else round(mtip[1], 1)},{'' if mtip is None else '스켈레톤'}")
        if a.max_frames and n >= a.max_frames: break
        if a.no_window: continue
        # ---- 화면 표시 ----
        vis = fr.copy()
        if show_mask and det.mask is not None:
            g = np.zeros_like(vis); g[..., 1] = 255
            vis = np.where(det.mask[..., None] > 0, cv2.addWeighted(vis, 0.6, g, 0.4, 0), vis)
        if det.entry is not None: cv2.circle(vis, (int(det.entry[0]), int(det.entry[1])), 10, (0, 220, 255), -1)
        if stip is not None:
            cv2.circle(vis, (int(stip[0]), int(stip[1])), 11, (0, 0, 255), -1); cv2.circle(vis, (int(stip[0]), int(stip[1])), 13, (255, 255, 255), 2)
        if mtip is not None: cv2.circle(vis, (int(mtip[0]), int(mtip[1])), 9, (255, 80, 0), -1)
        now = time.time(); fps_view = 0.9 * fps_view + 0.1 / max(1e-3, now - t_last); t_last = now
        state = ("기준 화면 찍는 중 (손 빼고 카메라 고정)" if det.ref is None else
                 "기준 화면이 안 맞아요 → 손 빼고 b" if det.bad else ("손끝 찾음" if stip is not None else "손 없음"))
        line = f"{state} | 색 차이 기준 {det.thr_ab:.0f} | 손 영역 {det.area}px | {fps_view:.0f} fps"
        if mp is not None: line += f" | MediaPipe: {'찾음' if mtip is not None else '못 찾음'}"
        vis = put_kr(vis, line, (10, 10), 24)
        vis = put_kr(vis, "빨강: 손끝(차이)  파랑: MediaPipe  노랑: 손이 들어온 쪽  |  b 기준 다시  t 틀림  [ ] 기준  m 영역  q 종료", (10, 44), 18)
        sc = min(1.0, 1400 / vis.shape[1], 900 / vis.shape[0])
        cv2.imshow("songil tip check", cv2.resize(vis, None, fx=sc, fy=sc))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        if k == ord("b"): det.ref = None; ref_buf = []; print("기준 화면 다시 찍는 중... 손을 빼 주세요")
        if k == ord("m"): show_mask = not show_mask
        if k == ord("["): det.thr_ab = max(3, det.thr_ab - 1)
        if k == ord("]"): det.thr_ab += 1
        if k == ord("t"):
            cv2.imwrite(os.path.join(a.outdir, "tip_wrong", f"wrong_{n_wrong:03d}_{t:.1f}s.jpg"), vis); n_wrong += 1; print(f"틀림 기록 {t:.1f}초")
    if writer: writer.release()
    open(os.path.join(a.outdir, "tips.csv"), "w", encoding="utf-8").write("\n".join(log))
    if mp is not None: open(os.path.join(a.outdir, "tips_mp.csv"), "w", encoding="utf-8").write("\n".join(log_mp))
    found = sum(1 for r in log[1:] if r.split(",")[1]); total = len(log) - 1
    msg = f"손끝(차이) 찾은 프레임 {found}/{total}"
    if mp is not None: msg += f", MediaPipe {sum(1 for r in log_mp[1:] if r.split(',')[1])}/{total}"
    print(msg); print("저장:", a.outdir, "(live_raw.mp4, tips.csv)")
    if not a.no_window: cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

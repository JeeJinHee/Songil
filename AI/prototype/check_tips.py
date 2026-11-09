"""손끝 인식이 맞는지 숫자로 확인하기
1) 정답 찍기:  python check_tips.py label --video kiosk1.mp4 --start 30 --end 48
   - 0.5초마다 프레임이 뜸. 손가락 끝을 마우스로 클릭 → 엔터(또는 스페이스)로 확정 → 다음 프레임
   - 손끝이 안 보이면 n, 잘못 찍었으면 다시 클릭, 그만하려면 q
   - 결과: labels_30-48.json
2) 비교:      python check_tips.py eval --labels labels_30-48.json --tips out\\01\\tips.csv
   - 방법별(스켈레톤/피부색/따라가기) 평균 오차와 20px·40px 안에 든 비율, 놓친 수, 헛잡은 수
"""
import argparse, json, csv, cv2, numpy as np, os

def label(a):
    cap = cv2.VideoCapture(a.video); out = f"labels_{a.start:g}-{a.end:g}.json"
    labels = json.load(open(out)) if os.path.exists(out) else {}
    click = {"p": None}
    def on(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN: click["p"] = (x, y)
    cv2.namedWindow("label"); cv2.setMouseCallback("label", on)
    for t in np.arange(a.start, a.end, a.step):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, fr = cap.read()
        if not ok: break
        s = min(1.0, 900 / fr.shape[0]); small = cv2.resize(fr, None, fx=s, fy=s); click["p"] = None
        while True:
            vis = small.copy()
            cv2.putText(vis, f"{t:.1f}s  click tip + Enter | n=no finger | q=quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if click["p"]: cv2.circle(vis, click["p"], 6, (0, 255, 0), -1)
            cv2.imshow("label", vis); k = cv2.waitKey(30) & 0xFF
            if click["p"] and k in (13, 32): break                    # 클릭 후 엔터/스페이스로 확정
            if click["p"] and k == 255: continue
            if k == ord("n"): click["p"] = None; break
            if k == ord("q"): json.dump(labels, open(out, "w")); print("저장:", out); return
        labels[f"{t:.3f}"] = None if click["p"] is None else [click["p"][0] / s, click["p"][1] / s]
    json.dump(labels, open(out, "w")); print("저장:", out, len(labels), "장")

def evaluate(a):
    labels = json.load(open(a.labels))
    rows = list(csv.DictReader(open(a.tips, encoding="utf-8")))
    ts = np.array([float(r["t"]) for r in rows])
    stats = {}; miss = false = 0; errs_all = []; dxy = []
    for k, gt in labels.items():
        r = rows[int(np.argmin(abs(ts - float(k))))]
        has = r["x"] != ""
        if gt is None:
            if has: false += 1
            continue
        if not has: miss += 1; continue
        e = float(np.hypot(float(r["x"]) - gt[0], float(r["y"]) - gt[1]))
        dxy.append((gt[0] - float(r["x"]), gt[1] - float(r["y"])))
        stats.setdefault(r["method"], []).append(e); errs_all.append(e)
    n_gt = sum(v is not None for v in labels.values())
    print(f"정답 프레임 {len(labels)}장 (손끝 보임 {n_gt}장)")
    print(f"놓침(손끝이 보이는데 못 찾음): {miss}   헛잡음(손끝이 없는데 찾음): {false}")
    for m, e in list(stats.items()) + [("전체", errs_all)]:
        e = np.array(e)
        if len(e): print(f"{m:<6} {len(e):>3}장  평균 오차 {e.mean():6.1f}px  중앙값 {np.median(e):6.1f}px  20px 이내 {100*(e<=20).mean():5.1f}%  40px 이내 {100*(e<=40).mean():5.1f}%")
    if dxy:
        d = np.array(dxy); mx, my = np.median(d[:, 0]), np.median(d[:, 1])
        rest = np.hypot(d[:, 0] - mx, d[:, 1] - my)
        print(f"치우침(정답 - 인식, 중앙값): 가로 {mx:+.1f}px, 세로 {my:+.1f}px  (+ 는 정답이 오른쪽/아래)")
        print(f"→ 추천 보정값: --tip-shift {mx:.0f},{my:.0f}   (보정하면 오차 중앙값 {np.median(np.hypot(d[:,0], d[:,1])):.1f}px → 약 {np.median(rest):.1f}px)")
        print("  ※ 카메라 위치·각도가 바뀌면 다시 재야 해요")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("label"); l.add_argument("--video", required=True); l.add_argument("--start", type=float, default=0)
    l.add_argument("--end", type=float, required=True); l.add_argument("--step", type=float, default=0.5)
    e = sub.add_parser("eval"); e.add_argument("--labels", required=True); e.add_argument("--tips", required=True)
    a = ap.parse_args(); label(a) if a.cmd == "label" else evaluate(a)

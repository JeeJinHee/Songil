"""정확도 차이를 한눈에 보여 주는 자료 만들기 (모델 다시 안 돌림, 좌표 파일만 사용 → 빠름)

python make_compare.py --video out\\cmp\\live_raw.mp4 --labels labels_5-25.json ^
       --base out\\cmp4\\tips_ugfd.csv --new out\\cmp4\\tips_ugfd2.csv --outdir out\\show

만들어지는 것 (out\\show)
  compare.mp4        왼쪽 '원본 공개 모델' / 오른쪽 '개선판' 나란히. 흰 십자 = 직접 찍은 정답,
                     빨간 선 = 정답까지 거리(오차, px). 아래에 지금까지의 평균 오차
  compare_chart.png  시간에 따른 오차 그래프 (두 모델) + 요약 숫자
  compare_best.png   차이가 가장 크게 나는 장면 3개를 나란히 (슬라이드용)

※ --base 는 보라(원본) 기록, --new 는 주황(개선판) 기록. 같은 영상으로 돌린 결과여야 함
※ 정답은 0.25초마다 찍은 점이라, 영상에서는 그 사이를 이어서(보간) 보여 줌. 그래프·숫자는 실제 찍은 점만 사용
"""
import argparse, csv, json, os, math
import cv2, numpy as np
from guide_sim import put_kr

C_BASE = (214, 79, 123)   # 보라 #7B4FD6 (BGR)
C_NEW = (0, 106, 194)     # 주황 #C26A00 (BGR)


def load_tips(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out[round(float(r["t"]), 3)] = None if r["x"] == "" else (float(r["x"]), float(r["y"]))
    return out


def nearest(d, t):
    k = min(d, key=lambda k: abs(k - t)); return d[k] if abs(k - t) < 0.02 else None


def gt_at(labels, t, max_gap=0.3):
    """정답을 시각 t에서: 찍은 점이면 그대로, 사이면 앞뒤 점을 이어서(보간). 앞뒤 중 하나라도 '손 없음'이면 None"""
    ks = sorted(labels)
    for k in ks:
        if abs(k - t) < 0.02: return labels[k], True
    prev = [k for k in ks if k < t]; nxt = [k for k in ks if k > t]
    if not prev or not nxt: return None, False
    a, b = prev[-1], nxt[0]
    if b - a > max_gap or labels[a] is None or labels[b] is None: return None, False
    w = (t - a) / (b - a); pa, pb = labels[a], labels[b]
    return (pa[0] + w * (pb[0] - pa[0]), pa[1] + w * (pb[1] - pa[1])), False


def stats(errs, miss):
    e = np.array(errs) if errs else np.array([np.nan])
    return dict(n=len(errs), miss=miss, med=float(np.nanmedian(e)), mean=float(np.nanmean(e)),
                in20=100 * float((e <= 20).mean()) if errs else 0, in40=100 * float((e <= 40).mean()) if errs else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--base", required=True, help="원본 공개 모델 기록 (tips_ugfd.csv)")
    ap.add_argument("--new", required=True, help="개선판 기록 (tips_ugfd2.csv)")
    ap.add_argument("--outdir", default=os.path.join("out", "show"))
    ap.add_argument("--start", type=float, help="영상 시작(초). 기본: 정답 구간 시작 - 0.5")
    ap.add_argument("--end", type=float, help="영상 끝(초). 기본: 정답 구간 끝 + 0.5")
    ap.add_argument("--crop", action="store_true", default=True, help=argparse.SUPPRESS)
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    labels = {float(k): v for k, v in json.load(open(a.labels)).items()}
    base, new = load_tips(a.base), load_tips(a.new)
    t0 = a.start if a.start is not None else max(0, min(labels) - 0.5); t1 = a.end if a.end is not None else max(labels) + 0.5

    # ---- 숫자 (실제 찍은 정답만) ----
    rows = []
    for k in sorted(labels):
        g = labels[k]
        if g is None: continue
        pb, pn = nearest(base, k), nearest(new, k)
        rows.append((k, None if pb is None else math.dist(pb, g), None if pn is None else math.dist(pn, g)))
    sb = stats([r[1] for r in rows if r[1] is not None], sum(r[1] is None for r in rows))
    sn = stats([r[2] for r in rows if r[2] is not None], sum(r[2] is None for r in rows))
    print(f"원본: 오차 중앙값 {sb['med']:.1f}px, 40px 이내 {sb['in40']:.0f}%, 놓침 {sb['miss']}")
    print(f"개선: 오차 중앙값 {sn['med']:.1f}px, 40px 이내 {sn['in40']:.0f}%, 놓침 {sn['miss']}")

    # ---- 그래프 ----
    make_chart(rows, sb, sn, os.path.join(a.outdir, "compare_chart.png"))

    # ---- 나란히 영상 ----
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, fr = cap.read()
    if not ok: raise SystemExit("영상을 열 수 없습니다")
    x0, x1 = content_cols(fr)                                   # 검은 여백 잘라서 화면 부분만
    W = x1 - x0; H = fr.shape[0]
    vw = cv2.VideoWriter(os.path.join(a.outdir, "compare.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * W + 20, H + 120))
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000); n = int(round(t0 * fps))
    run = {"b": [], "n": []}; frames_for_best = {}
    best_times = sorted([r for r in rows if r[1] is not None and r[2] is not None], key=lambda r: r[1] - r[2], reverse=True)[:3]
    best_set = {round(r[0], 3) for r in best_times}
    while True:
        ok, fr = cap.read()
        if not ok: break
        t = n / fps; n += 1
        if t > t1: break
        g, exact = gt_at(labels, t)
        panes = []
        for key, tips, color, title in (("b", base, C_BASE, "원본 공개 모델"), ("n", new, C_NEW, "개선판")):
            p = nearest(tips, round(t, 3)); v = fr[:, x0:x1].copy()
            if g is not None:
                gx, gy = int(g[0] - x0), int(g[1])
                if p is not None:
                    px, py = int(p[0] - x0), int(p[1]); e = math.dist(p, g)
                    cv2.line(v, (px, py), (gx, gy), (40, 40, 230), 3)
                    v = put_kr(v, f"{e:.0f}px", (min(W - 90, max(5, (px + gx) // 2 + 10)), max(5, (py + gy) // 2 - 14)), 24, (255, 255, 255), (40, 40, 230))
                    if exact: run[key].append(e)
                cv2.drawMarker(v, (gx, gy), (0, 0, 0), cv2.MARKER_CROSS, 40, 7); cv2.drawMarker(v, (gx, gy), (255, 255, 255), cv2.MARKER_CROSS, 40, 3)
            if p is not None:
                cv2.circle(v, (int(p[0] - x0), int(p[1])), 12, color, -1); cv2.circle(v, (int(p[0] - x0), int(p[1])), 14, (255, 255, 255), 2)
            elif g is not None:
                v = put_kr(v, "손끝 못 찾음", (10, H - 50), 26, (255, 255, 255), (40, 40, 40))
            v = put_kr(v, title, (10, 10), 32, (255, 255, 255), color)
            panes.append(v)
        canvas = np.full((H + 120, 2 * W + 20, 3), 255, np.uint8)
        canvas[:H, :W] = panes[0]; canvas[:H, W + 20:] = panes[1]
        for i, (key, color) in enumerate((("b", C_BASE), ("n", C_NEW))):
            e = run[key]; txt = f"지금까지 평균 오차 {np.mean(e):.0f}px" if e else "지금까지 평균 오차 -"
            canvas = put_kr(canvas, txt, (i * (W + 20) + 12, H + 14), 30, (30, 30, 30), (255, 255, 255))
        canvas = put_kr(canvas, "흰 십자 = 직접 찍은 정답 · 빨간 선 = 오차", (12, H + 70), 22, (90, 90, 90), (255, 255, 255))
        vw.write(canvas)
        if round(t, 3) in best_set: frames_for_best[round(t, 3)] = canvas.copy()
    vw.release()
    # ---- 차이 큰 장면 3개 ----
    shots = [frames_for_best[k] for k in sorted(frames_for_best)]
    if shots:
        hgt = 900; sm = [cv2.resize(s, (int(s.shape[1] * hgt / s.shape[0]), hgt)) for s in shots]
        sheet = np.full((hgt, sum(s.shape[1] for s in sm) + 30 * (len(sm) - 1), 3), 255, np.uint8); x = 0
        for s in sm: sheet[:, x:x + s.shape[1]] = s; x += s.shape[1] + 30
        cv2.imwrite(os.path.join(a.outdir, "compare_best.png"), sheet)
    print("저장:", a.outdir, "(compare.mp4, compare_chart.png, compare_best.png)")


def content_cols(fr):
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) > 16
    cols = np.nonzero(g.mean(0) > 0.05)[0]
    return (int(cols[0]), int(cols[-1]) + 1) if len(cols) > 10 else (0, fr.shape[1])


def make_chart(rows, sb, sn, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if any(f in x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(11, 7.6), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1.1], hspace=0.42)
    ax = fig.add_subplot(gs[0])
    ts = [r[0] for r in rows]
    for idx, color, name in ((1, "#7B4FD6", "원본 공개 모델"), (2, "#C26A00", "개선판")):
        ys = [r[idx] if r[idx] is not None else np.nan for r in rows]
        ax.plot(ts, ys, color=color, lw=2, marker="o", ms=4, label=name, zorder=3)
        miss = [r[0] for r in rows if r[idx] is None]
        if miss: ax.scatter(miss, [-6 if idx == 2 else -12] * len(miss), marker="x", color=color, s=40, zorder=4, clip_on=False)
        last = [(t, y) for t, y in zip(ts, ys) if not np.isnan(y)]
        if last: ax.annotate(name, last[-1], xytext=(6, 0), textcoords="offset points", color="#333333", fontsize=10, va="center")
    ax.axhline(50, color="#9aa0a6", lw=1, ls="--", zorder=1, label="참고: 메뉴 카드 절반 크기 (약 50px)")
    ax.set_xlabel("영상 시간 (초)"); ax.set_ylabel("정답과의 거리 (px, 낮을수록 정확)")
    any_miss = any(r[1] is None or r[2] is None for r in rows)
    ax.set_ylim(-16 if any_miss else 0, None); ax.grid(axis="y", color="#e8eaed", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    ax.set_title(f"손끝 위치 오차: 원본 공개 모델 vs 개선판  (직접 찍은 정답 {sb['n'] + sb['miss']}장 기준, × = 손끝 못 찾음)", loc="left", fontsize=12, color="#202124")
    # 요약 표
    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    lines = [("", "원본", "개선판"),
             ("오차 중앙값", f"{sb['med']:.1f}px", f"{sn['med']:.1f}px"),
             ("평균 오차", f"{sb['mean']:.1f}px", f"{sn['mean']:.1f}px"),
             ("20px 이내", f"{sb['in20']:.0f}%", f"{sn['in20']:.0f}%"),
             ("40px 이내", f"{sb['in40']:.0f}%", f"{sn['in40']:.0f}%"),
             ("놓침", f"{sb['miss']}장", f"{sn['miss']}장")]
    tb = ax2.table(cellText=[l[1:] for l in lines[1:]], rowLabels=[l[0] for l in lines[1:]], colLabels=lines[0][1:], loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(10.5); tb.scale(0.55, 1.5)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#dadce0")
        if r == 0: cell.set_text_props(color="#7B4FD6" if c == 0 else "#C26A00", weight="bold")
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    main()

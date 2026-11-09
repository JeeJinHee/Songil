# AI 프로토타입: 손끝 인식

노트북에서 휴대폰 카메라 영상(DroidCam 등)을 받아 **사용자의 검지 손끝 위치**를 찾는 실험 코드입니다.
Android 앱에 옮기기 전, 방식을 고르고 정확도를 재기 위한 용도입니다.

> 아직 안내 코드(`guide_sim.py`의 손끝 안내)와 **연결되지 않았습니다.** 녹화 영상으로만 평가했습니다.

## 결론 (11/7 기준)
- **공개 손끝 모델 개선판**을 사용 (MediaPipe·기준 화면 비교 방식은 제외)
- 직접 찍은 손끝 정답으로 평가: 오차 중앙값 **36.8px → 12.5px** (원본 공개 모델 대비)

| | 원본 공개 모델 | 개선판 |
|---|---|---|
| 오차 중앙값 | 36.8px | 12.5px |
| 평균 오차 | 62.9px | 22.9px |
| 40px 이내 | 52% | 77% |
| 놓침 (34장 중) | 1장 | 4장 |

## 파일
| 파일 | 역할 |
|---|---|
| `compare_tips.py` | 손끝 방식 비교 (주황 = 개선판, 보라 = 원본 공개 모델, 파랑 = MediaPipe, 빨강 = 기준 화면 비교) |
| `tip_check.py` | 기준 화면 비교 방식만 확인 (카메라 고정용) |
| `check_tips.py` | 손끝 정답 직접 표시(`label`) + 오차 계산(`eval`) |
| `make_compare.py` | 원본 vs 개선판 나란히 영상(`compare.mp4`)과 오차 그래프(`compare_chart.png`) 만들기 |
| `guide_sim.py`, `button_v0.py`, `screen.py` | 공통 함수 (카메라 열기, MediaPipe, 따라가기 등. 테스트 코드 v0.6) |
| `data/labels_fingertip_5-25.json` | 직접 찍은 손끝 정답 (0.25초 간격 80장) |

## 준비
### 1. 설치 (Python 3.13에서 확인)
```
pip install opencv-python mediapipe easyocr pillow numpy tensorflow matplotlib
```

### 2. 공개 손끝 모델 받기 (Git에 올리지 않음)
- 코드: https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection (MIT 라이선스)
  - SCUT-Ego-Gesture(1인칭 손 제스처 데이터)로 학습된 손 상자(YOLO) + 손끝 모델
- 가중치: 위 저장소 README의 Dropbox(또는 Mega) 링크에서 받기
- 받은 폴더 안에 `weights` 폴더를 만들고 두 파일을 넣기
  ```
  Unified-Gesture-and-Fingertip-Detection\weights\fingertip.h5
  Unified-Gesture-and-Fingertip-Detection\weights\yolo.h5
  ```
- 원래 코드는 TensorFlow 2.2 기준이지만 TensorFlow 2.21에서도 가중치를 불러올 수 있음을 확인함

## 실행
### 실시간 비교
```
python compare_tips.py --camera 1 --ugfd C:\songil\Unified-Gesture-and-Fingertip-Detection
```
- 시작 후 1~2초는 손을 화면에서 빼 두기 (기준 화면 찍기)
- 키: `o` 원본 켜기/끄기, `p`/`e`/`f` 개선판 기능 켜기/끄기, `r` 모델 입력 방향, `t` 틀린 장면 저장, `q` 종료
- 종료하면 `out\cmp\`에 `live_raw.mp4`(원본 영상)와 방식별 손끝 기록(`tips_*.csv`) 저장

### 녹화 영상으로 다시 돌리기 (같은 조건 비교)
```
python compare_tips.py --video out\cmp\live_raw.mp4 --outdir out\cmp2
```
- `--save-vis --labels 정답파일.json`: 점·정답 십자가 그려진 영상(`vis.mp4`) 저장

### 정확도 재기
```
python check_tips.py label --video out\cmp\live_raw.mp4 --start 5 --end 25 --step 0.25
python check_tips.py eval --labels labels_5-25.json --tips out\cmp2\tips_ugfd2.csv
```
- `label`: 영상이 한 장씩 뜨면 손톱 끝 클릭 → 엔터. 손이 없으면 `n`
- `eval`: 놓침, 헛잡음, 오차 중앙값·평균, 20px·40px 이내 비율

### 비교 자료 만들기
```
python make_compare.py --video out\cmp\live_raw.mp4 --labels labels_5-25.json --base out\cmp2\tips_ugfd.csv --new out\cmp2\tips_ugfd2.csv --outdir out\show
```

## 출력 파일 이름
| 파일 | 방식 |
|---|---|
| `tips.csv` | 기준 화면 비교 |
| `tips_mp.csv` | MediaPipe |
| `tips_ugfd.csv` | 공개 손끝 모델 원본 |
| `tips_ugfd2.csv` | **개선판** |
| `tips_ugfd3.csv` | 개선판 + 끝점 밀기 보정 (효과 없어 미채택) |

## 남은 문제
- 손이 빠르게 움직일 때 손끝보다 손가락 안쪽(40~60px)에 찍히는 장면
- 노트북 CPU에서 모델이 느림 (한 장 약 250ms) → 사이 프레임은 따라가기로 보완
- 영상 1개로만 평가함. 실제 키오스크·다른 조명에서 추가 확인 필요

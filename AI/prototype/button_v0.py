"""손길 버튼 인식 v0 (노트북 테스트용)
입력: 키오스크 촬영 프레임 이미지
출력: 버튼 목록 JSON + 디버그 이미지
단계: ① 화면 녹화 오버레이 제거 ② 화면 영역 추정 ③ OCR ④ 어두운 버튼·테두리 버튼 검출
      ⑤ 글자를 버튼에 배정 ⑥ 정보 표시 글자 걸러내기 ⑦ 선택 상태(빨간 테두리) 판정
"""
import cv2, numpy as np, json, re, sys, glob, os
import easyocr
from screen import overlay_mask

READER=None
USE_OVERLAY_MASK=True
FULL_FRAME=False        # True면 화면 영역을 찾지 않고 사진 전체에서 글자를 읽음 (--no-roi)   # 아이폰 화면 녹화본이면 True, 카메라 원본 영상이면 False
def reader():
    global READER
    if READER is None: READER=easyocr.Reader(['ko','en'],gpu=False,verbose=False)
    return READER

# ---------- 한글 자모 유사도 (OCR 오인식에 강한 비교) ----------
CHO="ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"; JUNG="ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"; JONG=" ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
def jamo(s):
    out=[]
    for ch in s.upper():
        c=ord(ch)
        if 0xAC00<=c<=0xD7A3:
            c-=0xAC00; out+= [CHO[c//588], JUNG[(c%588)//28]] + ([JONG[c%28]] if c%28 else [])
        elif ch.isalnum(): out.append(ch)
    return "".join(out)
def lev(a,b):
    if not a or not b: return max(len(a),len(b))
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(prev[j]+1,cur[j-1]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]
def sim(target,text):
    """target이 text 안에 부분적으로라도 들어있는 정도 (0~1)"""
    a,b=jamo(target),jamo(text)
    if not a or not b: return 0.0
    best=0.0
    for L in {len(a)-1,len(a),len(a)+1}:
        if L<=0: continue
        for i in range(0,max(1,len(b)-L+1)):
            best=max(best,1-lev(a,b[i:i+L])/len(a))
    return max(0.0,best)

# ---------- 규칙 (키오스크 일반 규칙) ----------
LINK_WORDS=["이전","다음","메인","확인","취소","전체 취소","결제","결제하기","선택완료","확대","한국어","English","주문하시려면 화면을 터치해주세요"]
INFO_PAT=[r"주문\s*수량",r"주문\s*금액",r"\(필수\)",r"\(선택\)",r"선택해\s*주",r"확인해\s*주",r"주문하세요",r"간편하게",r"^\s*[0-9.,]+\s*(원|개)?\s*$",r"^수량$",r"^가격$"]
OVERLAY_TEXT=["마이크 소리 꺼짐","탭하여 소리 켬"]   # 아이폰 화면 녹화 배너 (오인식돼도 걸러지게 유사도로 비교)

def screen_roi(img):
    h,w=img.shape[:2]
    if FULL_FRAME: return (0,0,w,h),[]
    ov=overlay_mask(img) if USE_OVERLAY_MASK else np.zeros((h,w),np.uint8)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    found=None
    for thr in (150,120):
        m=((hsv[...,2]>thr)&(hsv[...,1]<60)).astype(np.uint8)*255; m[ov>0]=0
        m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((9,9),np.uint8))
        m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((25,25),np.uint8))
        n,lab,st,_=cv2.connectedComponentsWithStats(m)
        if n<2: continue
        i=1+np.argmax(st[1:,4]); cand=st[i]
        if found is None or cand[4]>found[4]: found=cand
        if cand[4]>0.2*w*h and cand[2]<0.97*w: found=cand; break
    if found is None or found[4]<0.05*w*h:
        # 밝은 화면 영역을 못 찾음 (어두운 화면·다크 모드 등) → 사진 전체를 화면으로 보고 진행
        return (0,0,w,h),[]
    x,y,bw,bh,a=found
    y=max(0,y-int(0.06*h))   # 위쪽 어두운 카테고리 띠 포함
    cut=[s for s,c in (("left",x<=3),("right",x+bw>=w-3)) if c]
    return (int(x),int(y),int(x+bw),int(y+bh+int(0.01*h))),cut

def rect_candidates(img,roi):
    x0,y0,x1,y1=roi; sub=img[y0:y1,x0:x1]
    hsv=cv2.cvtColor(sub,cv2.COLOR_BGR2HSV)
    rects=[]
    # (a) 어둡게 채워진 버튼: 결제, 선택완료, 확인, 결제하기 등
    dark=((hsv[...,2]<110)&(hsv[...,1]<80)).astype(np.uint8)*255
    dark=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    for c in cv2.findContours(dark,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
        x,y,w,h=cv2.boundingRect(c); a=cv2.contourArea(c)
        if w*h<3000 or w*h>0.25*sub.shape[0]*sub.shape[1]: continue
        if a/(w*h)<0.75 or h>w*1.5: continue        # 소매·손은 사각형이 아님
        rects.append(dict(rect=[x+x0,y+y0,x+x0+w,y+y0+h],kind="filled"))
    # (b) 얇은 테두리 버튼: 옵션 버튼
    g=cv2.cvtColor(sub,cv2.COLOR_BGR2GRAY)
    e=cv2.Canny(cv2.GaussianBlur(g,(3,3),0),30,90)
    e=cv2.dilate(e,np.ones((3,3),np.uint8))
    for c in cv2.findContours(e,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]:
        x,y,w,h=cv2.boundingRect(c)
        if not(70<w<450 and 45<h<160): continue
        ap=cv2.approxPolyDP(c,0.04*cv2.arcLength(c,True),True)
        if len(ap)!=4 or cv2.contourArea(c)/(w*h)<0.7: continue
        rects.append(dict(rect=[x+x0,y+y0,x+x0+w,y+y0+h],kind="bordered"))
    # 중복 제거
    rects.sort(key=lambda r:(r["rect"][2]-r["rect"][0])*(r["rect"][3]-r["rect"][1]))
    keep=[]
    for r in rects:
        if any(iou(r["rect"],k["rect"])>0.5 for k in keep): continue
        keep.append(r)
    # 다른 버튼을 품고 있는 큰 사각형(옵션 줄 전체 테두리 등)은 버튼이 아님 → 버림
    def contains(o, i, pad=6): return o[0]-pad<=i[0] and o[1]-pad<=i[1] and i[2]<=o[2]+pad and i[3]<=o[3]+pad
    area=lambda r:(r[2]-r[0])*(r[3]-r[1])
    keep=[r for r in keep if not any(k is not r and contains(r["rect"],k["rect"]) and area(k["rect"])<0.8*area(r["rect"]) for k in keep)]
    return keep

def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=ix*iy; u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/u if u else 0
def center(b): return ((b[0]+b[2])/2,(b[1]+b[3])/2)
def inside(pt,r,pad=0): return r[0]-pad<=pt[0]<=r[2]+pad and r[1]-pad<=pt[1]<=r[3]+pad

def selected(img,r):
    x0,y0,x1,y1=r; sub=img[y0:y1,x0:x1]
    hsv=cv2.cvtColor(sub,cv2.COLOR_BGR2HSV)
    red=(((hsv[...,0]<10)|(hsv[...,0]>165))&(hsv[...,1]>90)&(hsv[...,2]>90)).mean()
    return bool(red>0.04)

def analyze(path,ocr=None):
    img=cv2.imread(path)
    roi,cut=screen_roi(img)
    if ocr is None:
        res=reader().readtext(img,mag_ratio=1.5,text_threshold=0.6,low_text=0.3)
        ocr=[{"box":[[int(x),int(y)] for x,y in b],"text":t,"conf":float(c)} for b,t,c in res]
    texts=[]
    for o in ocr:
        xs=[p[0] for p in o["box"]]; ys=[p[1] for p in o["box"]]
        b=[min(xs),min(ys),max(xs),max(ys)]
        if not inside(center(b),roi) or any(sim(k,o["text"])>=0.6 and len(jamo(o["text"]))>=0.6*len(jamo(k)) for k in OVERLAY_TEXT): continue
        texts.append(dict(box=b,text=o["text"].strip(" '\"`;:"),conf=o["conf"]))
    rects=rect_candidates(img,roi)
    buttons=[]; used=set()
    # 사각형 버튼에 글자 배정
    for r in rects:
        inner=[i for i,t in enumerate(texts) if inside(center(t["box"]),r["rect"],5)]
        if not inner: continue
        used.update(inner)
        name=" ".join(texts[i]["text"] for i in sorted(inner,key=lambda i:(texts[i]["box"][1]//20,texts[i]["box"][0])))
        conf=min(texts[i]["conf"] for i in inner)
        buttons.append(dict(name=name,rect=r["rect"],source=r["kind"],conf=round(conf,2),selected=selected(img,r["rect"])))
    # 남은 글자: 정보 표시 / 링크형 버튼 / 메뉴 카드(가격 아래) / 기타
    infos=[]
    for i,t in enumerate(texts):
        if i in used: continue
        s=t["text"]
        if len(jamo(s))<2 or t["conf"]<0.05: continue
        link=max(LINK_WORDS,key=lambda k:sim(k,s)); ls=sim(link,s)
        if ls>=0.8 and abs(len(jamo(link))-len(jamo(s)))<=3:
            buttons.append(dict(name=link,raw=s,rect=t["box"],source="text_link",conf=round(t["conf"],2),selected=False)); continue
        if any(re.search(p,s) for p in INFO_PAT): infos.append(s); continue
        buttons.append(dict(name=s,rect=t["box"],source="text_only",conf=round(t["conf"],2),selected=False))
    for k,b in enumerate(sorted(buttons,key=lambda b:(b["rect"][1]//40,b["rect"][0])),1): b["id"]=f"B-{k:02d}"
    buttons.sort(key=lambda b:b["id"])
    return img,dict(screen_roi=roi,screen_cut=cut,buttons=buttons,info_texts=infos)

COL={"filled":(0,140,255),"bordered":(255,120,0),"text_link":(0,170,0),"text_only":(160,160,160)}
def draw(img,res,target=None):
    vis=img.copy(); x0,y0,x1,y1=res["screen_roi"]; cv2.rectangle(vis,(x0,y0),(x1,y1),(255,0,255),3)
    for b in res["buttons"]:
        r=b["rect"]; c=COL[b["source"]]; th=8 if target and b["id"]==target else 3
        if target and b["id"]==target: c=(0,0,255)
        cv2.rectangle(vis,(r[0],r[1]),(r[2],r[3]),c,th)
        cv2.putText(vis,b["id"]+("*" if b["selected"] else ""),(r[0],max(20,r[1]-6)),cv2.FONT_HERSHEY_SIMPLEX,0.9,c,2)
    return vis

if __name__=="__main__":
    os.makedirs("out",exist_ok=True)
    for f in sys.argv[1:]:
        cache=f.replace("in/","out/").replace(".png","_ocr.json")
        ocr=json.load(open(cache)) if os.path.exists(cache) else None
        img,res=analyze(f,ocr)
        json.dump(res,open(f.replace("in/","out/").replace(".png","_buttons.json"),"w"),ensure_ascii=False,indent=1)
        cv2.imwrite(f.replace("in/","out/").replace(".png","_buttons.jpg"),cv2.resize(draw(img,res),None,fx=0.5,fy=0.5))
        print(os.path.basename(f),"cut=",res["screen_cut"])
        for b in res["buttons"]: print("  ",b["id"],b["source"],b["conf"],"[선택]" if b["selected"] else "",b["name"])

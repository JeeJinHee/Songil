import cv2, numpy as np, glob
def overlay_mask(img):
    """iOS 화면녹화 오버레이(마이크 배너, '나' 말풍선, 녹화 버튼) 영역. 원본 영상에는 없음."""
    h,w=img.shape[:2]; s=w/923
    m=np.zeros((h,w),np.uint8)
    for x0,y0,x1,y1 in [(225,125,700,255),(55,1765,235,1945),(755,215,905,365),(310,20,615,115)]:
        cv2.rectangle(m,(int(x0*s),int(y0*s)),(int(x1*s),int(y1*s)),255,-1)
    return m
def find_screen(img):
    h,w=img.shape[:2]
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    v=hsv[...,2]; sat=hsv[...,1]
    bright=((v>150)&(sat<60)).astype(np.uint8)*255
    bright[overlay_mask(img)>0]=0
    k=cv2.getStructuringElement(cv2.MORPH_RECT,(25,25))
    bright=cv2.morphologyEx(bright,cv2.MORPH_CLOSE,k)
    n,lab,st,_=cv2.connectedComponentsWithStats(bright)
    i=1+np.argmax(st[1:,cv2.CC_STAT_AREA])
    comp=(lab==i).astype(np.uint8)*255
    cnts,_=cv2.findContours(comp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    hull=cv2.convexHull(max(cnts,key=cv2.contourArea))
    return comp,hull
if __name__=="__main__":
    for f in sorted(glob.glob('in/*.png')):
        img=cv2.imread(f); comp,hull=find_screen(img)
        vis=img.copy(); cv2.drawContours(vis,[hull],-1,(0,0,255),8)
        cv2.imwrite(f.replace('in/','out/').replace('.png','_screen.jpg'),cv2.resize(vis,None,fx=0.25,fy=0.25))

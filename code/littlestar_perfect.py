from cv2 import cv2
import numpy as np
import os
import time

cap = cv2.VideoCapture(r"D:\project\video/sheep2.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
print(width, height)
lastFrame = cv2.imread(r"D:\project\image/find2.jpg")
lastFrame = cv2.cvtColor(lastFrame, cv2.COLOR_BGR2GRAY)


displayResize = 2000
FILE_OUTPUT = '/media/sf_VMshare/all.avi'
minArea = 6000

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(6,6))
kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))

clk=0
state=[]
start=[0]*11
over=[0]*11
final=[]
long=[0]*11

onex = 397
oney = 425
twox = 922
twoy = 423
threex = 397
threey = 478
fourx = 925
foury = 478

def key_detection(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,50,100)

    cv2.line(edges, (0,120), (1280,120), (255,255,255), 5)
    cv2.line(edges, (0,560), (1280,560), (255,255,255),5)
    dilation = cv2.dilate(edges, kernel1, iterations = 1)
    erosion = cv2.erode(dilation,kernel2,iterations = 1)
    cv2.imshow('erosion', erosion)
    key=[]
    
    contours1,hierarchy= cv2.findContours(erosion,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt1 in contours1:
        x,y,w,h = cv2.boundingRect(cnt1)
        area = cv2.contourArea(cnt1)
        digit=edges[y:y+h,x:x+w]

        if(area > 20000 and area < 60000):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,200,0),2)
            key.append([x,y,w,h])
        else:
            pass
    key.sort(key=lambda x: x[0])

    keyq=[[0]*3 for i in range (len(key))] 
    state=[0]*len(key) 
    keyy=len(key)
    for i in range(len(key)):
        for j in  range (3):
            keyq[i][j]=0
        state[i]=0


    num = 0
    letters = [ "c'", "d'", "e'", "f'", "g'", "a'", "b'", "c''", "d''", "e''", "f''", "g''","a''","b''","f", "g", "a", "b", "c'", "d'", "e'", "f'", "g'", "a'", "b'",
                "c''", "d''", "e''", "f''", "g''", "a''", "b''", "c'''", "d'''", "e'''", "f'''", "g'''", "a'''", "b'''",
                "c''''", "d''''", "e''''", "f''''", "g''''", "a''''", "b''''"]
    label = []
    for box in key:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, letters[num],(box[0],box[1]), font, 0.5,(0,0,255),2,cv2.LINE_AA)
        num += 1
    lastvalue = 0
    notes=[]
    for l in label:
        diff = abs(l[1] - lastvalue)
        if diff > 100:
            notes.append(l[0])
            cv2.putText(img, l[0],(450,100), font, 4,(255,0,0),5,cv2.LINE_AA)
            lastvalue = l[1]
    return img,key,state,keyy
   
def findContours(img,ori):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray_img) 
    blur_img = cv2.GaussianBlur(gray_img,(5,5),0)
    cv2.imshow("blur_name2", blur_img) 
    canny_img = cv2.Canny(blur_img,7,20)
    cv2.imshow("canny", canny_img) 
    img1 = cv2.dilate(canny_img, kernel3, iterations = 1)
    cv2.imshow("dilate", img1) 
    img2 = cv2.erode(img1,kernel4,iterations = 1)
    cv2.imshow("erode", img2) 
    contours, hierarchy = cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    press=[]
    for c in contours:
        if (60000> cv2.contourArea(c) > 10000):
            (x, y, w, h) = cv2.boundingRect(c)
            if(w<500):
                cv2.rectangle(ori, (x, y), (x + w, y + h ), (255, 255, 0), 2)
                press.append([x,y,w,h])
    press.sort(key=lambda x: x[0])
   
    return ori,press

def coordinate(coor):
    new=[]
    for i in coor:
       new.append([i[0],i[0]+i[2]])

    letters = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23]
    link=[]
    for j in range(len(new)):
        link.append([letters[j],new[j][0],new[j][1]])
    
    print("按鍵座標[按鍵序,按鍵左座標,按鍵右座標]:")
    print(link)
    
    return link

def centerpress(press):
    centerpoint = []
    for i in range(len(press)):
        centerpoint.append([press[i][0]+ (press[i][2]/2)])
    print("該幀數正被按下的琴鍵之中點座標:")
    print(centerpoint)
    
    return centerpoint

def centerpress(press):
    centerpoint = []
    for i in range(len(press)):
        centerpoint.append([press[i][0]+ (press[i][2]/2)])
    print("該幀數正被按下的琴鍵之中點座標:")
    print(centerpoint)
    
    return centerpoint

def note(link,centerpoint,state):
    k=[]
    for i in range(len(centerpoint)):
        for j in range(len(link)):  
            if (link[j][1]<centerpoint[i][0]<link[j][2]):
                k.append(link[j][0])

    print("該幀數正被按下的琴鍵之按鍵序:")
    print(k)
    
    for i in k:
        state[i]=1
    print("琴鍵之狀態（1為按下0為放開）:")
    print(state)

    return(k,state)

def starto(signal):

    for i in range(len(signal)):
        if signal[i]==1 and start[i]==0:
            start[i]=clk

    print("琴鍵被按下之起始幀數(CLK):")
    print(start)

    return start

def overdo(signal):
    letters = [ 'c', 'd', 'e', 'f', 'g', 'a', 'b', 'c\'', 'd\'', 'e\'', 'f\'', 'g\'', 'a\'', 'b\'',
                'c\'\'', 'd\'\'', 'e\'\'', 'f\'\'', 'g\'\'', 'a\'\'', 'b\'\'', 
                'c\'\'\'', 'd\'\'\'', 'e\'\'\'', 'f\'\'\'', 'g\'\'\'', 'a\'\'\'', 'b\'\'\'']
    m=0
    min=1000
    for i in range(len(signal)):
        if signal[i]==0 and start[i]!=0:
            over[i]=clk
            final.append([letters[i],start[i],over[i],over[i]-start[i]])
            start[i]=0

    for i in range(len(final)):
        if final[i][3]>m:
            m=final[i][3]
        if final[i][3]<min:
            min=final[i][3]
        
    print("琴鍵被放開之幀數(CLK):")
    print(over)
    print("按鍵紀錄[音名,起始幀數,結束幀數,持續長度]:")
    print(final)

    print("曾按下按鍵持續最久長度:")   
    print(m)
    print("曾按下按鍵持續最短長度:")   
    print(min)

    return final,m,min

def lilypond(notes):
    file = open('lily.ly', 'w')
    file.write(r'\\version  "2.22.0" \n \language "english" \n \\ relative{\n \clef "treble_8"')

    for i in notes:
        file.write(i)
        file.write("\n")
    file.write("}")
    file.close()
    
    return i

while True:
    ret, frame = cap.read()
    if not ret:
        break

    clk=clk+1
    print("====================================================================================")
    print("幀數(CLK):")
    print(clk) 

    
    detect = cv2.imread(r"D:\project\image/find2.jpg")
    key_img,coor,state,keyy=key_detection(detect) 
    cv2.imshow("key", key_img) 
    
    linkkk=coordinate(coor)
    cv2.line(frame, (0,120), (1280,120), (255,255,255), 5)
    cv2.line(frame, (0,560), (1280,560), (255,255,255),5)
    frame_project_2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask_diff = cv2.absdiff(frame_project_2, lastFrame)
    cv2.imshow("DIFF", fgmask_diff) 
    b = np.zeros(fgmask_diff.shape[:2], dtype = "uint8")
    r = np.zeros(fgmask_diff.shape[:2], dtype = "uint8")

    fgmask_diff_rgb = cv2.merge([b, fgmask_diff, r])
    fgmask_diff_rgb,press = findContours(fgmask_diff_rgb,frame)
    center=centerpress(press)
    cv2.imshow("final", fgmask_diff_rgb) 
    x1,x2=note(linkkk,center,state)
    starto(x2)
    final,m,min=overdo(x2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

x=['']*len(final)
y=''
final2=[]*clk

for i in range(clk):
    x=['']*len(final)
    y=''
    z=0
    w=0
    for j in range(len(final)):   
        if final[j][1]==i:
           x[j]=final[j][0]
           if w==0:
            y=y+x[j]
           elif w>0:
            y=y+' '+x[j]
           w=w+1
           if z==0:
               z=final[j][3]
    if y!='':
        final2.append([y,z,str(i),w])

for i in range(len(final2)):
    final2[i][1]=16/(final2[i][1]/min)
    if final2[i][1]>=12:
        final2[i][1]=16
    elif 6<=final2[i][1]<12:
        final2[i][1]=8
    elif 3<=final2[i][1]<6:
        final2[i][1]=4
    elif 2<=final2[i][1]<3:
        final2[i][1]=2
    elif 1<=final2[i][1]<2:
        final2[i][1]=1

print("按鍵紀錄整理[音名,幾分音符,起始幀數,同時按鍵數]:")
print(final2)

final3=['']*len(final2)
for i in range(len(final2)):
    if final2[i][3]==1:
        final3[i]=final2[i][0]+str(final2[i][1])
    else:
        final3[i]='<'+final2[i][0]+'>'+str(final2[i][1])

print("按鍵整理(lilypond格式):")
print(final3)
lilypond(final3)
from tkinter import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import requests
from bs4 import BeautifulSoup
import sys
import io
import webbrowser as w
from functools import partial
from tkinter import ttk
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns

def open_link(link):
    w.open(link)

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

url = "http://ncov.mohw.go.kr/"
url2 = "http://ncov.mohw.go.kr/bdBoardList_Real.do?brdId=1&brdGubun=12&ncvContSeq=&contSeq=&board_id=&gubun="

res = requests.get(url)
res2 = requests.get(url2)
res.raise_for_status() #문제 발생시 종료
res2.raise_for_status()
soup = BeautifulSoup(res.text, "lxml")
soup2 = BeautifulSoup(res2.text, "lxml")


window = Tk()
window.geometry("860x700")
window.configure(background='#F5ECCE')

a=soup.find("span", attrs={"class":"livedate"}).get_text()
현재날짜 = Label(window, text = a, bg='#F5ECCE').place(x=10, y=10)

#발생현황
a = soup.find("div", attrs={"class":"datalist"}).find_all("li")
p = a[0].find("span", attrs={"class":"data"}).get_text()
q = a[1].find("span", attrs={"class":"data"}).get_text()

국내발생 = Label(window, text = "국내발생", width=8, height=1, bg='#99CCFF').place(x=10, y=40)
국내발생수 = Label(window, text = p, fg='#FFFF00', bg='#99CCFF').place(x=70, y=40)
해외유입 = Label(window, text = "해외유입", width=8, height=1, bg='#99CCFF').place(x=100, y=40)
해외유입수 = Label(window, text = q, fg='#FFFF00', bg='#99CCFF').place(x=160, y=40)


#뉴스
tags=soup.select('body > div > div.mainlive_container > div.container > div > div.m_con_layout > div.m_newsarea > div.m_news > ul:nth-child(3)')[0].find_all("a")
뉴스 = Label(window, text="뉴스&이슈",fg='red', bg='#99CCFF').place(x=10, y=340)
뉴스1 = Button(window, text=tags[0].get_text(), bd=0, command=partial(open_link, 'http://ncov.mohw.go.kr'+tags[0]['href']), activebackground='black', activeforeground='white', bg='#E6E6E6').place(x=10, y=365)
뉴스2 = Button(window, text=tags[1].get_text(), bd=0, command=partial(open_link, 'http://ncov.mohw.go.kr'+tags[1]['href']), activebackground='black', activeforeground='white', bg='#E6E6E6').place(x=10, y=390)


#canvas=Canvas(window, relief="solid", bd=1, width=400, height=200).place(x=40, y=100)
# y4=Label(window, text="800", bg='#F5ECCE').place(x=10, y=100)
# y3=Label(window, text="600", bg='#F5ECCE').place(x=10, y=150)
# y2=Label(window, text="400", bg='#F5ECCE').place(x=10, y=200)
# y1=Label(window, text="200", bg='#F5ECCE').place(x=10, y=250)
df= pd.read_csv('list.csv')
df2= pd.read_csv('list2.csv')
list=df.loc['2021-05-29':]
list2=df2.loc['2021-05-29':]
figure = plt.Figure(figsize=(4,2), dpi=100)
ax2 = figure.add_subplot(111)
line = FigureCanvasTkAgg(figure,window)
line.get_tk_widget().place(x=40,y=100)
#sns.lineplot(x=list['Date'])
list.plot(kind='line', legend=True, ax=ax2, color='r', marker='o')
list2.plot(kind='line', legend=True, ax=ax2, color='b', marker='o')

# day1 = Label(window, text="532", bg='#F4FA58').place(x=50, y=300-(532/4))
# day2 = Label(window, text="480", bg='#F4FA58').place(x=107, y=300-(480/4))
# day3 = Label(window, text="430", bg='#F4FA58').place(x=164, y=300-(430/4))
# day4 = Label(window, text="459", bg='#F4FA58').place(x=221, y=300-(459/4))
# day5 = Label(window, text="677", bg='#F4FA58').place(x=278, y=300-(677/4))
# day6 = Label(window, text="681", bg='#F4FA58').place(x=335, y=300-(681/4))
# day7 = Label(window, text="695", bg='#F4FA58').place(x=392, y=300-(695/4))
#
# x1 = Label(window, text="6일전", bg='#F5ECCE').place(x=50, y=310)
# x1 = Label(window, text="5일전", bg='#F5ECCE').place(x=107, y=310)
# x1 = Label(window, text="4일전", bg='#F5ECCE').place(x=164, y=310)
# x1 = Label(window, text="3일전", bg='#F5ECCE').place(x=221, y=310)
# x1 = Label(window, text="2일전", bg='#F5ECCE').place(x=278, y=310)
# x1 = Label(window, text="1일전", bg='#F5ECCE').place(x=335, y=310)
# x1 = Label(window, text="오늘", bg='#F5ECCE').place(x=392, y=310)

#집단감염 발생지
집단감염발생지=Label(window, text="집단감염 발생지", bg='#99CCFF').place(x=10, y=450)
지역=Label(window, text="지역", bg='#F5ECCE').place(x=10, y=480)
주소=Label(window, text="주소", bg='#F5ECCE').place(x=120, y=480)
노출일자=Label(window, text="노출일자", bg='#F5ECCE').place(x=340, y=480)

c=0
tags=soup2.select('#content > div > div.box_line2 > div > div > table > tbody')[0].find_all("tr")
for tag in tags:
    지역=Label(window, text=tag.find('th').get_text(), bg='#F5ECCE').place(x=10, y=510+c)
    지역2=Label(window, text=tag.find_all('td')[0].get_text(), bg='#F5ECCE').place(x=65, y=510+c)
    주소=Message(window, text=tag.find_all('td')[2].get_text(), width=200, bg='#F5ECCE').place(x=120, y=510+c)
    노출일자=Label(window, text=tag.find_all('td')[3].get_text(), bg='#F5ECCE').place(x=340, y=510+c)
    c+=50


#지역별 거리두기 단계
step = soup.find("div", attrs={"class":"rss_detail"}).find_all("h4")
area = soup.find("div", attrs={"class":"rss_detail"}).find_all("p")

asd=''
for i in range(4):
    asd += '\n' + step[i].get_text() + '\n' + area[i].get_text().replace("\r\n\t","\n").strip() + '\n'

전국거리두기 = Label(window, text = '전국 거리두기 단계', bg='#99CCFF').place(x=480, y=40)
거리두기 = Label(window, text = asd, justify = 'left', width=40, height=22, bg='#F5ECCE').place(x=480, y=80)

단계별 = Label(window, text = '단계별 조치사항', bg='#99CCFF').place(x=480, y=450)
btn_url = 'http://ncov.mohw.go.kr/socdisBoardList.do?brdId=6&brdGubun=64&dataGubun=641'
btn_자세히 = Button(window, text = '(자세히)', bd=0, command = partial(open_link, btn_url), bg='#F5ECCE').place(x=680, y=450)


#거리두기 단계별 조치사항
qw=[]
qw.append([
    '카페, 음식점 운영 제한 해제\n2인 이상 매장 내 이용시 1시간\n이내로 강력 권고',
    '실내체육시설\n운영제한 해제',
    '유흥시설6종, 방문판매 운영시간\n제한(22시), 노래연습장, 파티룸, \n실내스탠딩공연장 음식섭취 금지,\n운영제한 해제',
    '영화관, PC방, 미용업,\n오락실, 독서실 등\n동반 자 외 좌석 한 칸 띄우기\n운영제한 해제'
])
qw.append([
    '카페, 음식점은 22시 이후\n포장·배달만 허용.\n2인 이상 매장 내 이용시\n1시간 이내로 강력 권고',
    '실내체육시설\n운영제한 해제(22시-05시)',
    '유흥시설6종, 방문판매,\n노래연습장, 파티룸,\n실내스탠딩공연장\n운영시간제한 (22시-05시)',
    '영화관, PC방, 미용업,\n오락실, 독서실 등\n음식섭취 금지 및\n좌석 한 칸 띄우기\n운영제한 해제'
])
qw.append([
    '카페, 음식점은 22시 이후\n포장·배달만 허용.\n2인 이상 매장 내 이용시\n1시간 이내로 강력 권고',
    '실내체육시설 21시-05시 \n운영중단, 격렬한운동 (GX, 줌바, \n스피닝, 에어로빅) 금지, \n스크린골프 룸당 4명 허용, \n샤워실 운영금지 (수영장제외)',
    '유흥시설6종 집합금지,\n방문판매 등 직접판매 홍보관,\n노래연습장 등 음식 섭취 금지.\n21시 이후 운영 중단',
    '영화관, PC방, 미용업,\n오락실, 독서실 등\n21시 이후 운영 중단 추가'
])
qw.append([
    '카페는 포장·배달만 허용,\n음식점은 21시 이후로\n포장·배달만 허용\n시설 면적 8㎡ 당 \n1명까지로 인원 제한',
    '산업·생활에 필수적인 시설\n(필수산업시설, 거주 숙박시설,\n음식점류, 상점류, 장사시설, \n의료시설) 외에는 집합금지',
    '산업·생활에 필수적인 시설\n(필수산업시설, 거주 숙박시설,\n음식점류, 상점류, 장사시설, \n의료시설) 외에는 집합금지',
    '산업·생활에 필수적인 시설\n외에는 집합금지'
])

def event1(event):
    if comboBox.get() == '1.5단계':
        event2(0)
    elif comboBox.get() == '2단계':
        event2(1)
    elif comboBox.get() == '2.5단계':
        event2(2)
    elif comboBox.get() == '3단계':
        event2(3)

def event2(i):
    qwe1 = Label(window, text = qw[i][0], width=25, height=6, bg='#F5ECCE', relief="groove").place(x=480, y=480)
    qwe2 = Label(window, text = qw[i][1], width=25, height=6, bg='#F5ECCE', relief="groove").place(x=665, y=480)
    qwe3 = Label(window, text = qw[i][2], width=25, height=6, bg='#F5ECCE', relief="groove").place(x=480, y=580)
    qwe4 = Label(window, text = qw[i][3], width=25, height=6, bg='#F5ECCE', relief="groove").place(x=665, y=580)

comboBox = ttk.Combobox(window, width=8, state='readonly')
comboBox['values'] = ('1.5단계', '2단계', '2.5단계', '3단계')
comboBox.grid(column=0, row=0)
comboBox.place(x=580, y=450)
comboBox.current(0)
comboBox.bind('<<ComboboxSelected>>',event1)


window.mainloop()

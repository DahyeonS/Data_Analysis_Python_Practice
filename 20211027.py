# -*- coding: utf-8 -*-
"""

-Part.0 오늘은 뭘 할까요?-

1. 주제 선정
2. 스크립트 제작
3. 작동 확인
"""

"""
-Chap.0 개관-

1. 몇 달만에 고민했는데 일단 이것부터 시작을 하는 게 좋을 것 같습니다.
2. 아주아주 기초적인 작업임
3. 먼저 복습부터 시작합니다.


import urllib.request as req # 원격 서버 url 자료 요청
from bs4 import BeautifulSoup # source -> html 파싱

url = "http://media.daum.net"

# 1. url 요청
res = req.urlopen(url)
# object.method()
src = res.read() # source
print(src) #\xb1\x85\xec\x9e: 한글 깨짐

# 디코딩 적용
data = src.decode('utf-8') # 디코딩 적용
print(data)

# 2. html 파싱
html = BeautifulSoup(data, 'html.parser')
print(html)
'''
<태그 속성='값'> 내용 </태그>
<a herf='www.naver.com'> 네이버 </a>
'''

# 3. tag 요소 추출
'''
select('tag[속성='값']')
'''

# 1) tag element 수집
a_tag = html.select('a[class="link_txt"]')
'''
html.select_one()
html.select()
'''
len(a_tag) # 62

a_tag[0]
type(a_tag[0]) # bs4.element.Tag

# 2) 자료 수집
crawling_data = [] # news 저장

cnt = 0
for a in a_tag : 
    cont = str(a.string) # 내용 가져오기 -> 문자열
    print(cnt, cont, sep=' -> ')
    cnt += 1
    crawling_data.append(cont.strip())
    # str.strip(): 문단 끝 불용어(\n,\t\r 공백) 제거
    
# 46 -> 코스피

print(crawling_data)

crawling_data = crawling_data[:46] # 코스피, 바로잡기 삭제

print(crawling_data)




-Chap.1 뉴스제목&썸네일로 문서 만들기-

목적: 실물로 출력가능한, 인터넷 게시글에 업로드 가능한 형태의
     뉴스 기사 헤드라인과 사진을 크롤링하여 정리
     
사용 기술: 웹 크롤링, 사용 패키지: BeautifulSoup

대상 데이터: 다음 뉴스
네이버의 경우 크롤링에 적대적인 정책을 펼치므로 난이도가 높다고 판단,
다음 뉴스 데이터으로만 작업합니다.
"""

import urllib.request as req # 원격 서버 url 자료 요청
from bs4 import BeautifulSoup # source -> html 파싱

url = "http://media.daum.net"

# 1. url 요청
res = req.urlopen(url)
src = res.read() # source

# 2. 디코딩 적용
data = src.decode('utf-8') # 디코딩 적용

# 3. html 파싱
html = BeautifulSoup(data, 'html.parser')

# 4. tag element 수집
a_tag = html.select('a[class="link_txt"]')

# 5. 자료 수집
crawling_data = [] # news 저장

cnt = 0
for a in a_tag : 
    cont = str(a.string) # 내용 가져오기 -> 문자열
    print(cnt, cont, sep=' -> ')
    cnt += 1
    crawling_data.append(cont.strip())
    # str.strip(): 문단 끝 불용어(\n,\t\r 공백) 제거

crawling_data = crawling_data[:46] # 코스피, 바로잡기 삭제
print(crawling_data)

'''
# 6. 정리


# 7. 이미지 추출
img_tag = html.select('img')
crawling_img = [] # 이미지 저장

cnt = 0
for i in img_tag : 
    cont = str(i.string) # 내용 가져오기 -> 문자열
    print(cnt, cont, sep=' -> ')
    cnt += 1
    crawling_data.append(cont.strip())
    # str.strip(): 문단 끝 불용어(\n,\t\r 공백) 제거

crawling_data = crawling_data[:46] # 코스피, 바로잡기 삭제
print(crawling_data)
'''
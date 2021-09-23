# NamuWiki-recommender
나무위키 문서 추천 프로젝트입니다. 크롬 확장프로그램을 이용하여 사용자가 최근 읽은 문서를 바탕으로 사용자에게 적합한 문서를 추천합니다.
문서 추천으로는 Doc2Vec을 이용하고 있습니다.

## 사용 방법
### 확장 프로그램 이용자
프로그램 작동 방식과는 상관없이 확장프로그램을 이용하실 분들은 https://kmikey1004.tistory.com/4 를 참고해주세요.

### 데이터 정제
문서 추천기는 나무위키 덤프데이터를 이용합니다. https://namu.wiki/w/%EB%82%98%EB%AC%B4%EC%9C%84%ED%82%A4:%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4%20%EB%8D%A4%ED%94%84 에서 먼저 덤프 데이터를 다운받으시길바랍니다.
이후 덤프데이터를 json_parser 을 이용하여 정제해야합니다.
```
python json_dump.py --json_path="" --output=""
```
json_parser로 파싱한 데이터를 그대로 학습에 사용할 수 없습니다. preprocess 를 이용하여 추가로 데이터를 정제해주어야합니다.
```
python preprocess.py --csv_path="" 
```
json_parser는 json 형태로 되어있는 덤프데이터를 마크다운을 제거하여 파싱해오고, preprocess는 redirection을 제거하며 user-item pair의 형태로 데이터를 정제합니다.

이후 데이터를 doc2vec 을 이용하여 학습합니다. doc2vec은 태그를 이용하는 방식과 문서의 내용을 이용하는 방식 두 가지 모델을 사용합니다.

## Requirements
- tensorflow 2.6.0
- gensim 4.1.1
- transformer 4.9.2


## TODO
- 불필요한 코드 정리
- 구글 드라이브를 이용한 설치가 아닌 스토어에서 설치
- 추천기 성능 향상

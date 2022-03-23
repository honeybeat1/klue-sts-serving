# preonboarding_klue_sts
프리온보딩 [KLUE](https://github.com/KLUE-benchmark/KLUE) STS task를 위한 REST API 서버 코드입니다.  

🏠 Install
---
이 GitHub 저장소를 클론하여 설치하세요. 

     git clone https://github.com/honeybeat1/preonboarding_klue_sts.git
     pip install -r requirements.txt

🔧 Build with
---
- FastAPI 0.75.0
- pyTorch 1.6.0
- Transformers 4.15.0


🏃‍♀️ API 실행
---
로컬 IP address(http://127.0.0.1:8000/docs) 로 접속합니다.docs를 통해 FastAPI GUI로 접근하여 
`POST` method를 통해 비교하고자 하는 Sentence1, Sentence2를 json형식으로 전송합니다.  

<img src="https://github.com/honeybeat1/preonboarding_klue_sts/blob/6961e7402db55205d1f17f1ce302ac781ba2de54/api_req_res.png" width="600">


📞 Response
---

    {
      'sentence1': 첫번째 문장, 
      'sentence2': 두번째 문장, 
      'Similarity[0-5]': 유사도(dataset의 real-label로 0-5 scale), 
      'Is it Similar?': Yes, No(threshold:3.0)
    }

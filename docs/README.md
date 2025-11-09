# 3DAnimals Documentation

이 폴더는 프로젝트 개발 중 발견된 문제, 해결 방법, 그리고 Best Practice를 문서화합니다.

## 📁 폴더 구조

```
docs/
├── README.md                    (이 파일)
├── reports/                     버그 분석 및 기술 보고서
│   └── 20251109_arti_params_none_error.md
├── guides/                      개발 가이드 및 Best Practices
│   ├── progressive_training_best_practices.md
│   └── coding_defensive_programming.md
└── troubleshooting/             문제 해결 가이드 (향후 추가)
```

## 📚 문서 목록

### 보고서 (Reports)

| 날짜 | 제목 | 설명 |
|------|------|------|
| 2025-11-09 | [arti_params None Reference 버그](reports/20251109_arti_params_none_error.md) | Progressive Training 환경에서 발생한 RuntimeError 분석 및 해결 |

### 가이드 (Guides)

| 제목 | 설명 | 대상 |
|------|------|------|
| [Progressive Training Best Practices](guides/progressive_training_best_practices.md) | Progressive Training 환경에서 안전한 코드 작성 방법 | 개발자 |
| [Defensive Programming Guide](guides/coding_defensive_programming.md) | Python 방어적 프로그래밍 기법 | 모든 개발자 |

## 🎯 문서 작성 가이드라인

### 보고서 (Reports) 작성 시

**파일명 형식:** `YYYYMMDD_brief_description.md`

**포함 내용:**
- 날짜, 버전, 작성자
- Executive Summary
- 문제 상황 및 에러 로그
- 근본 원인 분석
- 해결 방법
- 영향 범위
- 학습 포인트

### 가이드 (Guides) 작성 시

**파일명 형식:** `topic_name.md`

**포함 내용:**
- 대상 독자
- 목차
- 개념 설명
- 예제 코드 (Good/Bad 비교)
- 체크리스트
- 참고 자료

## 🔍 문서 검색

### 주제별 색인

**Progressive Training:**
- [Progressive Training Best Practices](guides/progressive_training_best_practices.md)
- [arti_params 버그 분석](reports/20251109_arti_params_none_error.md)

**코드 품질:**
- [Defensive Programming](guides/coding_defensive_programming.md)

**버그 수정:**
- [arti_params None Reference](reports/20251109_arti_params_none_error.md)

## 📝 문서 기여 방법

1. **버그 발견 및 수정 시:**
   - `docs/reports/`에 상세 분석 보고서 작성
   - 날짜를 포함한 파일명 사용
   - Git commit에 문서 링크 포함

2. **Best Practice 발견 시:**
   - `docs/guides/`에 가이드 문서 작성
   - 실제 예제 포함
   - 기존 문서 업데이트

3. **문제 해결 팁:**
   - `docs/troubleshooting/`에 FAQ 형식으로 작성 (향후)

## 🏷️ 버전 관리

각 문서는 하단에 변경 이력 테이블 포함:

| 날짜 | 버전 | 변경 내용 | 작성자 |
|------|------|----------|--------|
| YYYY-MM-DD | 1.0 | 초안 작성 | Name |

## 📞 문의

문서 내용에 대한 질문이나 개선 제안은 GitHub Issue로 등록해주세요.

---

**마지막 업데이트:** 2025-11-09

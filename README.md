# CLAP Evaluation And Analysis Pipeline (UIQ)

본 프로젝트는 CLAP 계열 모델의 추론 결과를 일관된 방식으로 분석하기 위한 파이프라인입니다.

현재 워크플로는 아래처럼 고정됩니다.

1. run_eval 단계: 모델 추론 수행, raw 결과 JSONL 저장
2. eval 단계: raw 결과를 하나의 중간 캐시 eval.jsonl로 병합
3. analysis 단계: 모든 분석을 중간 캐시 기반으로 수행하고 analysis_output에 저장

즉, 분석은 temp 리포트나 수동 요약 파일이 아니라, 항상 results의 순수 모델 출력(JSONL)에서 시작합니다.

## 핵심 특징

- 모델별 raw 출력 스키마 일관 사용
- 중간 캐시 eval.jsonl 재사용으로 반복 분석 속도 향상
- 분석 설정을 config.yaml의 analysis 섹션으로 통합
- level 1, level 2, heatmap, disagreement, uniformity를 단일 엔트리에서 재현 가능

## 프로젝트 구조

```text
.
├── config.yaml
├── pyproject.toml
├── scripts/
│   ├── run_eval.py
│   └── eval.py
├── results/
│   └── eval_outputs/
│       ├── laion_train_results.jsonl
│       ├── laion_test_results.jsonl
│       ├── m2d_train_results.jsonl
│       ├── m2d_test_results.jsonl
│       ├── mga_train_results.jsonl
│       ├── mga_test_results.jsonl
│       ├── msclap_train_results.jsonl
│       └── msclap_test_results.jsonl
├── analysis_output/
│   ├── eval.jsonl
│   ├── category_uniformity_*.csv
│   ├── margin_heatmap_*.png
│   ├── margin_heatmap_tables.json
│   ├── disagreement_matrix_*.png
│   └── disagreement_matrix_tables.json
└── src/clap_eval/
    ├── pipeline.py
    ├── config.py
    └── analysis/
        ├── result_io.py
        ├── category_uniformity.py
        ├── margin_heatmap.py
        └── disagreement_matrix.py
```

## 실행 순서

### 1) 환경 동기화

```bash
uv sync
```

### 2) 모델 추론

```bash
uv run python scripts/run_eval.py --config config.yaml
```

이 단계 결과는 results/eval_outputs 폴더에 모델별 split별 JSONL로 저장됩니다.

### 3) 분석 전체 실행

```bash
uv run python scripts/eval.py --config config.yaml
```

mode를 명시하면 개별 분석만 실행할 수 있습니다.

```bash
uv run python scripts/eval.py uniformity --config config.yaml
uv run python scripts/eval.py margin-heatmap --config config.yaml
uv run python scripts/eval.py disagreement --config config.yaml
```

## config.yaml 설정

자주 바꾸는 분석 설정은 analysis 섹션에서 관리합니다.

```yaml
analysis:
  input_dir: results/eval_outputs
  output_dir: analysis_output
  cache_filename: eval.jsonl
  cmap: RdBu
  max_samples_per_category: 2000
  model_order: [laion, m2d, mga, msclap]
  force_rebuild_cache: false
```

설명:

- input_dir: 모델 raw JSONL 위치
- output_dir: 분석 결과 저장 위치
- cache_filename: 병합 캐시 파일명
- cmap: heatmap 색상맵
- max_samples_per_category: uniformity 계산 시 카테고리 샘플 상한
- model_order: 시각화 열 순서
- force_rebuild_cache: true면 캐시를 항상 재생성

## 레벨 정의

- level 1: 메타 도메인 레벨
  - Human, Animal, Nature, Music, Machine_Vehicle, Tool_Mechanism, Home_Everyday, Sports_Action, Other
- level 2: 세부 캡션 레벨
  - VGGSound 세부 클래스 단위

## 분석 파일별 의미 (논문 작성용)

### 중간 캐시

- analysis_output/eval.jsonl
  - 원본 raw 결과를 병합한 단일 분석 입력
  - 필드 예시: model, split, index, youtube_id, start_time, label, embedding, text_embedding, meta_domain, sample_id
  - 목적: 반복 분석 시 디스크 탐색과 파싱 비용 절감

### Uniformity 분해

- analysis_output/category_uniformity_full.csv
  - 모델 x 레벨 x 카테고리 단위 uniformity 전체 결과
  - 핵심 수식:
    - Uniformity = log E[exp(-t ||f(x)-f(y)||^2)]
    - 각 카테고리 내부 샘플 쌍에 대해 계산
  - 컬럼:
    - model, level, category, sample_count, uniformity, rank_within_model

- analysis_output/category_uniformity_top20_level1.csv
- analysis_output/category_uniformity_top20_level2.csv
  - 모델별로 붕괴 가능성이 큰 카테고리 상위 20개 요약
  - 기본 정렬은 uniformity 값 기준

### Margin Heatmap

- analysis_output/margin_heatmap_level1.png
- analysis_output/margin_heatmap_level2_all.png
- analysis_output/margin_heatmap_level2_negative_only.png
- analysis_output/margin_heatmap_tables.json

계산 맥락:

1. 카테고리 내부에서 오디오 임베딩과 텍스트 임베딩 cosine similarity 행렬 구성
2. positive similarity는 대각 성분
3. max negative similarity는 같은 행에서 대각 제외 최대값
4. margin = positive - max_negative
5. 카테고리별 평균 margin을 모델별로 집계하여 heatmap 생성

해석 가이드:

- 평균 margin < 0: negative가 positive를 이긴 구간, semantic collapse 가능성 증가
- 평균 margin > 0: 상대적으로 구분이 유지된 구간

### Disagreement Matrix

- analysis_output/disagreement_matrix_level1.png
- analysis_output/disagreement_matrix_level2.png
- analysis_output/unique_success_matrix_level1.png
- analysis_output/disagreement_matrix_tables.json

계산 맥락:

1. sample_id 기준으로 모델 간 같은 샘플 정렬
2. 각 모델의 sample margin 부호를 비교
3. pairwise disagreement 카운트
   - 예: A fail and B pass = count(margin_A < 0 and margin_B > 0)
4. unique success 카운트
   - only A pass = count(margin_A > 0 and others <= 0)

해석 가이드:

- 특정 도메인에서 A fail and B pass가 크면 B가 해당 도메인을 더 방어
- unique success가 0이어도 이상은 아님
  - 조건이 매우 엄격해서 pairwise disagreement는 존재하되 unique는 0일 수 있음

## 재현성 참고

- run_eval output 스키마는 pipeline.py 기준으로 유지됩니다.
- 분석은 항상 raw output에서 시작하므로 temp 중간 파일 의존이 없습니다.
- cache는 force_rebuild_cache false일 때 최신 raw 파일보다 오래된 경우에만 재생성됩니다.

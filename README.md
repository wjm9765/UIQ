# CLAP Evaluation Pipeline (UIQ)

오디오-텍스트 대조 학습 모델(CLAP류)의 Zero-shot 성능 평가 및 표현 붕괴(Representation Collapse) 분석을 위한 파이프라인입니다. LAION-CLAP, MGA-CLAP, M2D-CLAP 등의 모델을 VGGSound 등의 데이터셋을 활용해 빠르고 메모리 효율적으로 평가할 수 있도록 구성되어 있습니다.

## 🌟 주요 기능

- **다중 모델 단일 파이프라인 평가**: LAION, MGA, M2D 등 다양한 구조의 CLAP 모델을 동일한 환경(VGGSound)에서 일관되게 평가합니다.
- **메모리 효율적 처리 (OOM 방지)**: 데이터셋 전체를 메모리에 올리지 않고 Generator 기반으로 순차적 인퍼런스를 수행하여 OOM을 방지합니다.
- **OS 호환 환경 구축**: 초고속 패키지 매니저 `uv`를 통해 `pyproject.toml`에 명시된 의존성을 동기화하며, Linux용 CUDA 패키지와 MacOS용 패키지를 자동으로 분기하여 설치합니다.
- **다목적 분석 리포트 제공**:
  1. 검색 정확도 메트릭 (R@1, R@5, R@10, MRR)
  2. 임베딩 분리력 / 붕괴도 분석 (Intra vs Inter Similarity, Margin)
  3. 모델별 최약점 클래스 도출 (Vulnerable Categories Top-5)

---

## 📂 프로젝트 구조

```plaintext
├── config.yaml               # 전체 실행/평가 설정 파일 (수정 시 최우선)
├── pyproject.toml            # Python 패키지 및 환경 설정 파일 (uv 지원)
├── README.md                 # 프로젝트 가이드
├── results/                  # 모델 추론 결과물(jsonl) 기본 저장 디렉토리
├── input/                    # HF 모델/데이터셋 캐시 저장소
├── src/
│   └── clap_eval/            # 핵심 파이프라인 모듈 패키지
│       ├── config.py         # Config 파싱 로직
│       ├── dataset.py        # HF VGGSound 등 데이터셋 로더
│       ├── pipeline.py       # 인퍼런스 파이프라인 전개
│       └── models/           # 각 CLAP 모델별 Wrapper (laion, mga, m2d)
└── scripts/
    ├── setup_models.py       # 로컬 가중치 파일 등 다운로드
    ├── run_eval.py           # 파이프라인 실행, 인퍼런스 후 JSONL 결과 생성
    └── evaluate_all_claps.py # 생성된 JSONL을 분석하여 평가 리포트 출력
```

---

## 🚀 시작하기 (Getting Started)

### 1. 환경 설정 (Dependencies)
초고속 패키지 매니저 [`uv`](https://github.com/astral-sh/uv)를 사용합니다.

```bash
# 의존성 설치 및 동기화
# (macOS와 Linux(CUDA)를 자동으로 구분하여 올바른 버전 설치)
uv sync


### 2. 가중치 준비
MGA, M2D 등 로컬 스토리지에 체크포인트가 필요한 모델들의 파라미터를 허깅페이스에서 스크립트를 통해 다운받습니다.
```bash
./scripts/setup_models.py
```

---

## 💡 코드 실행 방법

이 프로젝트의 모든 주 기능 및 구동은 `./scripts/` 디렉토리 내부의 스크립트들을 통해 실행됩니다.

### Step 1. 인퍼런스 파이프라인 수행 (JSONL 생성)
`config.yaml`에 정의된 데이터셋, 스트리밍 설정, 하이퍼파라미터를 읽고 입력 오디오의 모델 임베딩을 추론합니다. 완료되면 결과를 텍스트 파일(jsonl)로 저장합니다.

```bash
./scripts/run_eval.py --config config.yaml
# 또는 python scripts/run_eval.py --config config.yaml
```
> **참고:** 실행이 완료되면 `config.yaml` 안의 `execution.output_dir` (기본값: `results/eval_outputs`)에 `[모델이름]_results.jsonl` 형태의 파일이 생성됩니다. 이 파일은 각 데이터의 인덱스, 임베딩, 정답 라벨 등 필요한 모든 정보를 갖습니다.

### Step 2. 결과 평가 및 분석 (리포트 생성)
도출된 `.jsonl` 파일을 스크립트가 다시 읽어들여 객관식 보기 풀(Pool)을 생성하고, 모델 간 메트릭 비교 평가 및 붕괴 분석을 수행합니다.

```bash
./scripts/evaluate_all_claps.py
# 또는 python scripts/evaluate_all_claps.py
```
> **참고:** 평가 스크립트는 `config.yaml`의 **`evaluation.results_dir`** 경로를 읽어 자동으로 데이터를 가져옵니다. 경로를 바꿔 평가하고 싶다면 yaml 설정 파일만 변경하시면 됩니다!

---

## ⚙️ Configuration (`config.yaml`)
직접 코드를 하드코딩할 필요 없이 대부분의 설정 제어는 `config.yaml`로 가능합니다.

- **`dataset:`**
  - 평가할 HuggingFace 데이터셋, 로컬 다운로드 및 메모리 설정(`streaming` 옵션), 카테고리당 샘플링 수 제어
- **`models:`**
  - 개별 모델들의 활성화 스위치(`enabled: true/false`), 로컬 및 원격 가중치 경로 설정
- **`execution:`**
  - 하드웨어 리소스 전략 (batch size, device 설정)
  - `output_dir`: 인퍼런스 완료 파일 생성 폴더 지정
- **`evaluation:`**
  - `results_dir`: 평가 시 읽어들일 대상 데이터 폴더 (예: `eval_outputs_sample`)
  - `top_k_list`: `R@K` 출력 단위 지정

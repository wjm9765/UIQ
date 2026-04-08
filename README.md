# CLAP Models Evaluation Pipeline on VGGSound

VGGSound 데이터셋을 기반으로 다양한 **CLAP 모델(LAION, MGA, M2D)** 의 성능(예: 제로샷 성능 저하 및 카테고리별 차이)을 평가하기 위한 객체 지향, 확장성 중심의 통합 파이프라인입니다.

각 모델을 동시에 로드하지 않고 순차적으로(Sequential) 로드 및 언로드함으로써 한정된 GPU 메모리를 극대화(`배치 최대화`)할 수 있도록 설계되었습니다.

---

## 📁 디렉토리 구조 (Directory Structure)

```text
UIQ/
├── .gitignore              # 외부 데이터(체크포인트, 다운로드 리포지토리, 임시결과 등) 무시 설정
├── config.yaml             # 전체 파이프라인 제어 파일 (데이터셋 양, 모델 경로, 메모리 전략 등)
├── pyproject.toml          # uv 설정 및 패키지 의존성 파일
├── README.md               # 프로젝트 개요 및 실행 가이드 (이 문서)
├── input/                  # 다운로드된 데이터 보관 폴더
│   ├── audio/              # 원본 VGGSound .wav 오디오 파일 디렉토리
│   └── vggsound.csv        # VGGSound 원본 Metadata CSV
├── results/
│   └── eval_outputs/       # 모델별 Inference(임베딩 결과) JSONL 저장소
├── models_third_party/     # 허깅페이스에 없는 스탠드얼론(Standalone) Clone 리포지토리 폴더 
│   ├── MGA-CLAP/           # MGA 모델 소스 코드
│   └── m2d/                # M2D 모델 소스 코드
├── checkpoints/            # 깃허브에서 개별 다운로드한 가중치 파일(.pt, .pth) 보관 폴더
├── scripts/                # 파이프라인 실행 스크립트 (실행 엔트리포인트)
│   ├── setup_server.sh     # GPU 리눅스 서버 초기 구축용 bash 스크립트 (ffmpeg 및 uv 설치)
│   ├── setup_vggsound.py   # 초기 VGGSound CSV 메타데이터 파일 구조 생성 및 다운로드
│   ├── setup_models.py     # 외부 딥러닝 깃허브 리포지토리 Clone 및 환경 준비
│   └── run_eval.py         # 실제 모델 평가를 실행하는 메인 루프 스크립트
└── src/
    └── clap_eval/
        ├── __init__.py
        ├── config.py       # config.yaml의 Python 매핑 클래스
        ├── dataset.py      # 카테고리당 사용할 샘플 개수를 필터링하고 경로를 내어주는 모듈
        ├── pipeline.py     # 모델 개별 로드부터 결과 생성까지 통합 관리
        └── models/         # 추상화된 모델 평가 래퍼 (Wrapper)
            ├── base.py     # BaseClapModel 추상 클래스
            ├── laion.py    # LAION CLAP (HuggingFace Native) 
            ├── mga.py      # MGA CLAP (Local Codebase)
            └── m2d.py      # M2D CLAP (Local Codebase)
```

---

## 🚀 실행 순서 (Quick Start Guide)

### Step 0: 리눅스/GPU 서버 초기 세팅 (Linux/Ubuntu 사용자 전용)
만약 비어있는 GPU 서버 인스턴스를 처음 발급받으셨다면, 인메모리 오디오 스트리밍에 필요한 `ffmpeg`와 파이썬 패키지 매니저 `uv` 설치를 자동화해주는 쉘 스크립트를 가장 먼저 실행하세요. (이미 세팅된 로컬 Mac/PC라면 건너뛰시면 됩니다.)
```bash
./scripts/setup_server.sh
```
> 내부적으로 `apt-get install ffmpeg`, `pip install uv`, 그리고 `uv sync`까지 자동으로 모두 진행됩니다.

### Step 1: 파이썬 패키지 의존성 동기화 (`uv sync`)
(앞선 Step 0을 실행했다면 이 과정은 생략해도 됩니다)
`pyproject.toml`에 명시된 라이브러리(PyTorch, Torchaudio, Transformers, Accelerate, yt-dlp 등)를 설치하고 현재 패키지를 인식시킵니다.
```bash
uv sync
```

### Step 2: MGA & M2D 모델 소스 다운로드
허깅페이스 공식 지원에 없는 `M2D` 및 `MGA-CLAP` 모델의 오픈소스 환경을 구성합니다.
스크립트를 실행하면 `models_third_party` 폴더 안에 GitHub 프로젝트가 클론되고, 가중치를 넣을 `checkpoints` 폴더가 생성됩니다.
```bash
./scripts/setup_models.py
```

### Step 3: 체크포인트 가중치 및 오디오 원본 수동 이동
> **이 단계는 수동으로 진행해야 합니다.** 파일 사이즈 문제로 자동화에 한계가 있습니다.

1. `MGA` 및 `M2D` 오픈소스 개발자들의 구글 드라이브, 홈페이지에서 공개된 오리지널 **가중치 파일(`.pt`, `.pth` 등)** 을 다운받아 `checkpoints/` 안에 넣습니다.
2. 다운로드 완료된 `.wav` 파일들을 `input/audio/` 안에 규칙에 맞게 (`{youtube_id}_{start_time:06d}.wav`) 집어넣습니다.

### Step 4: 메타데이터 CSV 다운로드 
`vggsound.csv` 구조를 분석하고 샘플링을 진행하려면 원본 CSV가 필요합니다. 스크립트를 실행해 `input/vggsound.csv` 를 생성합니다.
```bash
./scripts/setup_vggsound.py
```

### Step 5: (Option) Local Model Wrapper 코드 완성
가중치 배치가 완료되었다면 `src/clap_eval/models/mga.py` 와 `m2d.py`의 `_load_model()` 메서드 안에, **실제 깃허브 내 모델 Class를 Import하는 로직**을 알맞게 매핑해줍니다. (이미 `sys.path` 설정 로직은 뼈대에 짜여져 있습니다)

### Step 6: 모델 평가 및 임베딩 추출 실행
모든 준비가 끝나면 파이프라인을 실행합니다. 카테고리당 뽑을 샘플 수, 작동할 모델 목록은 `config.yaml` 안에서 직접 켜고(`enabled: true/false`), 수량을 조절할 수 있습니다.
```bash
./scripts/run_eval.py --config config.yaml
```

* 평가 실행 시 지정된 `output_dir` (기본: `results/eval_outputs/`) 에 `[model_name]_results.jsonl` 형태로 각 라벨에 따른 Audio 임베딩 및 정답, 예측값 정보가 기록됩니다. 이후, 추출된 벡터를 가지고 별도 Metric 분석 코드를 구동시키면 됩니다.

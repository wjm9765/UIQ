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

## � 60GB 데이터셋의 메모리/디스크 핸들링 원리
이 프로젝트는 **Hugging Face Datasets**를 사용하여 거대한 VGGSound 60GB 오디오 데이터를 통째로 다운로드하거나 메모리에 올리는 부담을 방지하고, 유연하게 제어합니다. `config.yaml`의 `streaming` 플래그로 두 가지 동작 모드를 지원합니다.

* **`streaming: true` (개발/테스트 환경용)**
  * 하드 디스크 여유 공간이 없을 때 사용합니다.
  * Hugging Face 서버에서 오디오 배열(Array)을 실시간으로 가져와 바로 GPU로 던집니다. 디스크 공간을 사용하지 않지만, 네트워크 속도에 따라 조금 느릴 수 있습니다.
* **`streaming: false` (실제 1TB 프로덕션 환경용 - 강력 권장 🚀)**
  * **60GB 전체를 RAM(메모리)에 올리지 않습니다! 절대 OOM(메모리 부족)이 나지 않습니다.**
  * Hugging Face가 최초 실행 시 전체 60GB 데이터셋을 `config.yaml`에 지정해 둔 로컬 경로(`input/`)에 고속 Arrow/Parquet 바이너리 포맷으로 안전하게 다운로드하여 보관합니다.
  * 이후 파이프라인에서 읽어들일 때는, 전체를 메모리에 올리는 대신 **OS 레벨의 메모리 매핑(Memory-mapping)** 기술을 사용하여 모델이 요청하는 배치(Batch) 만큼만 순식간에 RAM으로 퍼올립니다. 속도가 압도적으로 빠르고 매우 안정적입니다.

---

## 🚀 실행 순서 (Quick Start Guide)

### Step 0: 리눅스/GPU 서버 초기 세팅 및 동기화 (`uv sync`)
이 파이프라인은 엄격하고 빠른 패키지 의존성 관리 환경인 `uv`를 사용합니다. PyTorch, Torchaudio, Transformers, HuggingFace Datasets 모듈을 동기화합니다. 비어있는 GPU 인스턴스라면 환경부터 구성합니다.
```bash
./scripts/setup_server.sh
uv sync
```

### Step 1: MGA & M2D 모델 소스 및 가중치 자동 다운로드
허깅페이스 공식 지원에 없는 `M2D` 및 `MGA-CLAP` 모델의 오픈소스 환경을 구성하고, 지정된 프라이빗 허깅페이스 데이터셋(`wjm9765/clap_weights`)에서 가중치를 다운로드합니다. 
```bash
export HF_TOKEN="본인의_허깅페이스_토큰"
./scripts/setup_models.py
```
> 실행 완료 시 모델 깃허브 코드가 Clone되고, `checkpoints/` 폴더 내에 가중치 모델링 파일들이 자동 저장됩니다.

### Step 2: 설정 파일 수정 (`config.yaml`)
현재 환경에 맞게 `config.yaml`을 튜닝합니다. 실제 프로덕션 서버로 옮길 땐 데이터셋 파라미터만 바꾸시면 가장 안전하고 오작동 없는 환경이 구성됩니다.
```yaml
dataset:
  name: "VGGSound"
  hf_repo: "txya900619/vggsound-16k"
  cache_dir: "input"       # 전체 다운로드 시 저장될 로컬 캐시 폴더 경로 (기본값)
  streaming: true          # 개발/테스트 시 true, 1TB 프로덕션 서버에서는 반드시 false로 변경 (가장 빠름!)
  samples_per_class: 100   # 실제 실험 시 100, 수량 확인 및 테스트 시 2등의 값으로 조절
```

### Step 3: 메인 평가 파이프라인 실행
모든 준비가 끝나면 파이프라인을 구동하여 허깅페이스 데이터셋에서 오디오를 48kHz로 가져와 즉시 임베딩을 평가합니다. (더 이상 수동 CSV 다운로드나 유튜브 스크래핑이 작동하지 않습니다).
```bash
uv run scripts/run_eval.py --config config.yaml
```

* 평가 실행 시 지정된 `output_dir` (기본: `results/eval_outputs/`) 에 `[model_name]_results.jsonl` 형태로 각 라벨에 따른 Audio 임베딩 및 정답, 예측값 정보가 기록됩니다. 이후, 추출된 벡터를 가지고 별도 Metric 분석 코드를 구동시키면 됩니다.

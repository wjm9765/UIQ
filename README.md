# CLAP Models Evaluation Pipeline on VGGSound

VGGSound 데이터셋을 기반으로 다양한 **CLAP 모델(LAION, MGA, M2D)** 의 성능(예: 제로샷 성능 저하 및 카테고리별 차이)을 평가하기 위한 통합 파이프라인입니다.

---

## 📁 디렉토리 구조 (Directory Structure)

```text
UIQ/
├── .gitignore              # 외부 데이터(체크포인트, 다운로드 리포지토리, 임시결과 등) 무시 설정
├── config.yaml             # 전체 파이프라인 제어 파일
├── config_laion.yaml       # 개별 독립 실행용 제어 파일
├── config_m2d.yaml         # 개별 독립 실행용 제어 파일
├── config_mga.yaml         # 개별 독립 실행용 제어 파일
├── pyproject.toml          # uv 설정 및 패키지 의존성 파일
├── README.md               # 프로젝트 개요 및 실행 가이드
├── input/                  # 다운로드된 데이터 보관 폴더
├── results/
│   └── eval_outputs/       # 모델별 Inference(임베딩 결과) JSONL 저장소
├── models_third_party/     # 허깅페이스에 없는 3rd-party 모델 레포지토리
│   ├── MGA-CLAP/           # MGA 모델 소스 코드
│   └── m2d/                # M2D 모델 소스 코드
├── checkpoints/            # 깃허브에서 개별 다운로드한 다운로드 가중치 파일(.pt, .pth)
├── scripts/                # 파이프라인 보조 스크립트
├── run_laion.sh            # 단일 모델 병렬 평가 실행 스크립트 (LAION)
├── run_m2d.sh              # 단일 모델 병렬 평가 실행 스크립트 (M2D)
├── run_mga.sh              # 단일 모델 병렬 평가 실행 스크립트 (MGA)
└── src/
    └── clap_eval/
        ├── pipeline.py     # 16kHz 단위의 순수 원본 오디오 단일 Dataloader
        └── models/         # 추상화된 모델 평가 래퍼 (Wrapper)
            ├── laion.py    # LAION CLAP (HuggingFace Native - 48kHz GPU 자체 리샘플링) 
            ├── mga.py      # MGA CLAP (Local Codebase - 32kHz GPU 리샘플링, 10초 잘라내기 적용)
            └── m2d.py      # M2D CLAP (Local Codebase - 16kHz 원본 크기 유지, 10초 패딩/잘라내기 적용)
```

---

## 🔧 파이프라인 주요 변경 및 최적화 사항

* **독립된 16kHz 고정 Dataloader Pipeline:** 공통 오디오 파이프라인이 여러 모델에게 연이어 데이터를 평가하더라도 앞선 48kHz 리샘플링이 다음 모델에게 강요되지 않습니다. Dataloader는 언제나 원본 16kHz 오디오만을 전달하며, 리샘플링(`torchaudio.transforms.Resample`)은 각 모델 래퍼(`laion.py`, `mga.py`, `m2d.py`) 내부에서 GPU 위에서 1회만 단독 수행되어 성능 병목을 예방합니다.
* **배치 단위 오디오 제어(Tansform & Truncate):** 모델 구조의 한계(예: MGA HTSAT 모델의 Swin 아키텍처)에 의한 고정 버퍼 길이 초과 오류(`AssertionError: the wav size should less than or equal to the swin input size`)를 미연에 방지하고자, 10초를 초과하거나 모자라는 길이의 오디오는 모델 래퍼가 스스로 자르거나(Truncate) 부족하면 0으로 채우도록(Padding) 패치되었습니다.
* **완벽한 GPU 이전 최적화(Device transfer):** 기존 외부 소스코드의 파편화로 인해 자체 스펙트로그램 레이어 등의 객체가 CPU 메모리에 도태되어 남아있으면서 발생하던 텐서 기종 충돌(`Input type and weight type should be the same`) 문제가 해결되어 전체 서브 모델 모듈 트리들이 `cuda`에 정상적으로 통째로 로드됩니다.
* **신규 의존성 충돌 무결점 완화:** 구버전 API 호환성 유지를 위해 `datasets<4.0.0` 등은 물론이고 `sed_scores_eval==0.0.0`, `setuptools<70.0.0` 제약을 걸어 두었으며 API가 폐기된 `ruamel.yaml`의 함수(`safe_load()`) 사용 로직을 `YAML(typ='safe')` 구문 및 `strict=False` 체크 무효화 등으로 강제 우회하여 구동되도록 모든 최신 의존성 에러 악재를 걷어냈습니다.

---

## 🚀 멀티 터미널을 활용한 3-Way 단일 서버 병렬 실행 (속도 3배 증가)

기본적으로 `config.yaml` 1개로 3개의 모델을 직렬(`sequential`)로 돌릴 수 있으나, VRAM 용량이 크더라도 단일 프로세스가 처리하는 CPU 병목(Data I/O Bottleneck)으로 인해 A100 등 대형 GPU의 점유율이 20% 수준에 머무는 문제가 있습니다. 

이를 해결하기 위해 모델별로 설정 파일을 3개로 분할하였으므로, VS Code 터미널 탭을 **3개로 나란히 여신 뒤 아래의 개별 스크립트를 하나씩 병렬로 띄우시면** CPU를 훨씬 골고루 쓰면서 단일 서버 안에서 세 모델 평가를 월등한 속도로 마칠 수 있습니다.

### Terminal 1:
```bash
./run_laion.sh
```

### Terminal 2:
```bash
./run_m2d.sh
```

### Terminal 3:
```bash
./run_mga.sh
```

* 위 과정을 통해 단일 프로세스로 인퍼런스 하는 시간 제약을 줄일 수 있으며, 모든 파이프라인의 평가는 `results/eval_outputs/` 폴더 내에 `[model_name]_results.jsonl` 형태로 기록됩니다. (.jsonl 안에는 Audio 임베딩 및 Caption 텍스트 기반 Text 임베딩 값이 모조리 기록되어 있습니다.)

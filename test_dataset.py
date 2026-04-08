#!/bin/bash
#SBATCH --job-name=vgg_download     # 작업 이름
#SBATCH --output=vgg_download.out   # 정상 출력 로그가 저장될 파일
#SBATCH --error=vgg_download.err    # 에러 로그가 저장될 파일
#SBATCH --cpus-per-task=4           # 데이터 전처리/다운로드에 사용할 CPU 코어 수
#SBATCH --mem=32G                   # 할당받을 메모리 용량
#SBATCH --time=24:00:00             # 최대 작업 시간 (24시간)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=master,n01,n02

# 디스크 안전장치: 남은 용량이 100GB 미만이면 작업 취소 (60GB 다운로드 대비)
AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')

if [ "$AVAILABLE_SPACE" -lt 100 ]; then
    echo "🚨 Error: 디스크 잔여 용량이 ${AVAILABLE_SPACE}GB밖에 없습니다. (최소 100GB 필요) 작업을 중단합니다."
    exit 1
fi

echo "디스크 용량 확인 완료 (${AVAILABLE_SPACE}GB 남음). 다운로드를 시작합니다."

# 파이썬 스크립트 실행
python prepare_hf_vgg.py
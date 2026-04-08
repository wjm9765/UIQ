#!/bin/bash
#SBATCH --job-name=eval          # 작업 이름 (평가 작업에 맞게 변경)
#SBATCH --output=slurm_output/eval_base_%j.out
#SBATCH --error=slurm_output/eval_base_%j.err
#SBATCH --cpus-per-task=4             # CPU 코어 수
#SBATCH --mem=32G                     # 메모리
#SBATCH --time=24:00:00               # 최대 작업 시간
#SBATCH --nodes=1                     # 사용할 노드 수
#SBATCH --ntasks-per-node=1           # 노드당 태스크 수
#SBATCH --gres=gpu:1                  # 💡 텍스트 임베딩 추출을 위해 GPU 1개 할당 필수!
#SBATCH --exclude=master,n01,n02      # 지정된 노드 제외 (n03 등 계산 노드에 배정됨)

export HF_TOKEN=""
export PYTHONPATH=$PYTHONPATH:/data/kyongminkong/Cap/UIQ

echo "================================================="
echo "작업 시작 시간: $(date)"
echo "할당된 계산 노드: $SLURM_NODELIST"
echo "================================================="

# 1. 프로젝트 최상단 디렉토리로 이동
cd /data/kyongminkong/Cap/UIQ

# 2. 마스터 노드의 환경 변수와 경로 설정을 그대로 가져오기
source ~/.bashrc


# 3. 평가 스크립트 실행
/data/kyongminkong/.conda/envs/mfds2/bin/python scripts/evaluate_all_claps.py

echo "================================================="
echo "작업 종료 시간: $(date)"
echo "================================================="
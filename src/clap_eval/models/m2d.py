import torch
import numpy as np
import sys
from pathlib import Path
from .base import BaseClapModel

class M2DClapModel(BaseClapModel):
    def _load_model(self):
        import sys
        from transformers import RobertaTokenizer, RobertaModel
        
        print(f"Loading M2D-CLAP from: {self.checkpoint_path}")
        
        # 1. 외부 Repo 경로 추가
        if self.repo_path:
            repo_abs_path = str(Path(self.repo_path).resolve())
            if repo_abs_path not in sys.path:
                sys.path.insert(0, repo_abs_path)

        try:
            # 2. 텍스트 인코더 및 토크나이저 준비
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.text_encoder = RobertaModel.from_pretrained("roberta-base").to(self.device)
            
            # 3. 오디오 인코더 준비 (M2D 레포지토리의 build_model 함수 호출 필요)
            # ⚠️ 아래 import는 m2d 레포 구조에 따라 'from model import m2d_vit' 등으로 바뀔 수 있습니다.
            from m2d_model import m2d_vit_base as m2d_model 
            self.audio_model = m2d_model().to(self.device)

            # 4. 가중치 로드 (checkpoint-30.pth)
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            
            # 가중치 파일 내의 state_dict 키 명칭을 확인해야 합니다. ('model' 또는 'state_dict')
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            self.audio_model.load_state_dict(state_dict, strict=False)
            
            self.audio_model.eval()
            self.text_encoder.eval()
            print("✅ M2D-CLAP 모델 및 RoBERTa 인코더 로드 완료!")
            
        except Exception as e:
            print(f"❌ M2D 로드 실패: {e}")

    @torch.no_grad()
    def get_audio_embedding(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        # M2D models typically use 16kHz
        target_sr = 16000
        import librosa
        
        if len(audio_data.shape) > 1 and audio_data.shape[0] > 1:
            audio_data = librosa.to_mono(audio_data)
            
        if sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
            
        # Expected to implement custom processing and feature extraction 
        return np.zeros((1, 512)) # dummy shape

    @torch.no_grad()
    def get_text_embedding(self, texts: list[str]) -> np.ndarray:
        # 1. 토크나이징 (프롬프트 추가 권장)
        prompts = [f"This is a sound of {t}" for t in texts]
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        
        # 2. RoBERTa 추론
        outputs = self.text_encoder(**inputs)
        
        # 3. 보통 [CLS] 토큰의 임베딩이나 평균값을 사용합니다.
        # M2D-CLAP 설정에 따라 pooler_output 또는 last_hidden_state[:, 0] 사용
        embeddings = outputs.pooler_output 
        
        # 4. L2 정규화 (유사도 계산을 위해 필수!)
        embeddings = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy()

    @torch.no_grad()
    def get_audio_embedding(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        return np.zeros((1, 768)) # dummy shape

    @torch.no_grad()
    def get_text_embedding(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 768)) # dummy shape

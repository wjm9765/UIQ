import io
import yt_dlp
import ffmpeg
import numpy as np

def stream_youtube_audio_memory(youtube_id: str, start_time: int, duration: int = 10, target_sr: int = 48000):
    """
    유튜브 URL에서 오디오 스트림만 가져와 ffmpeg를 통해 메모리 상의 numpy 배열로 변환하는 함수.
    로컬 디스크에 .wav 파일을 저장하지 않고 Lazy 로딩합니다.
    
    :param youtube_id: 유튜브 비디오 ID
    :param start_time: 시작 시간(초)
    :param duration: 잘라낼 길이(초) - VGGSound는 기본 10초
    :param target_sr: 오디오 샘플링 레이트
    :return: (audio_array, sr) 또는 실패 시 (None, None)
    """
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    
    # yt-dlp 설정 (다운로드 스킵, 포맷은 최고 음질 추출)
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'nocheckcertificate': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            stream_url = info.get('url', None)
            
        if not stream_url:
            return None, None

        # ffmpeg로 필요한 구간(start_time ~ start_time+duration)만 잘라내서 파이프로 출력
        out, _ = (
            ffmpeg
            .input(stream_url, ss=start_time, t=duration)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=target_sr)
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        
        # 바이트 데이터를 numpy 배열(단정밀도 float32)로 곧바로 변환
        audio_data = np.frombuffer(out, np.float32)
        
        if len(audio_data) == 0:
            return None, None
            
        return audio_data, target_sr
        
    except Exception as e:
        print(f"\n[STREAM ERROR] {youtube_id} ({start_time}s) 가져오기 실패: {str(e)}")
        return None, None

다음 작업을 해주세요:

1. 먼저 seopbo의 github 저장소(https://github.com/seopbo/nemo-skills-patch) 링크를 확인해서 최신 상태의 패치가 맞는지 확인하세요.
2. 그 다음 `workspace/lmalign/nemo-skills-harness/setup.sh` 파일을 실행하세요.
2. 아래 파이썬 코드를 실행해서 nltk의 'punkt_tab' 리소스를 다운로드 해주세요:

   ```
   python3 -c "import nltk; nltk.download('punkt_tab')"
   ```

위 두 단계를 모두 수행해야 합니다.

# /opt/benchmarks 생성 (권한 필요할 수 있음)
sudo mkdir -p /opt/benchmarks
sudo chown "$USER:$USER" /opt/benchmarks

# IFBench 클론
cd /opt/benchmarks
git clone https://github.com/allenai/IFBench.git

# 의존성 설치 (저장소의 requirements.txt 사용)
cd IFBench
pip install -r requirements.txt
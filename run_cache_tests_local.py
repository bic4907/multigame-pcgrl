#!/usr/bin/env python3
"""
로컬에서 multigame 캐시 테스트를 실행하는 Python 스크립트
GitHub Actions의 multigame-cache-tests.yml 워크플로우와 동일한 테스트를 수행합니다
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Python 버전 확인"""
    print("1. Python 버전 확인...")
    version = sys.version
    print(f"   {version}")
    print()


def check_and_install_dependencies():
    """필요한 패키지 확인/설치"""
    print("2. 필요한 패키지 확인...")
    required_packages = [
        ("numpy", "numpy"),
        ("pillow", "PIL"),
        ("pytest", "pytest"),
    ]

    for package_name, import_name in required_packages:
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', '알 수 없음')
            print(f"   ✓ {package_name} ({version}) 설치됨")
        except ImportError:
            print(f"   ✗ {package_name} 설치 필요")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    print()

def check_cache_status():
    """캐시 상태 확인"""
    print("3. 캐시 상태 확인...")
    cache_dir = Path("dataset/multigame/cache/artifacts")
    
    if cache_dir.exists():
        # 캐시 디렉토리 크기 계산
        total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        print(f"   ✓ 캐시 디렉토리 존재: {cache_dir} ({size_mb:.1f} MB)")
    else:
        print(f"   ⚠ 캐시 디렉토리 없음: {cache_dir}")
        print("   테스트 중에 생성될 것입니다.")
    print()

def run_test_1():
    """기본 정규화 및 캐시 유틸 테스트"""
    print("4. Base Normalization & Cache Utils 테스트 실행...")
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "dataset/multigame/tests/test_base_normalization.py",
            "dataset/multigame/tests/test_cache_utils.py",
            "-q", "--override-ini=testpaths="
        ],
        cwd=os.getcwd()
    )
    print()
    return result.returncode == 0

def run_test_2():
    """타일 매핑 및 렌더 검증 테스트"""
    print("5. Tile Mapping & Render Validation 테스트 실행...")
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "dataset/multigame/tests/test_tile_utils.py",
            "-v", "--override-ini=testpaths=",
            "-k", "TestMappedShape or TestMappedMinMax or TestMappedRender or TestEndToEnd"
        ],
        cwd=os.getcwd()
    )
    print()
    return result.returncode == 0

def main():
    """메인 실행 함수"""
    print("=" * 35)
    print("Multigame Cache Tests - Local Run")
    print("=" * 35)
    print()
    
    try:
        check_python_version()
        check_and_install_dependencies()
        check_cache_status()
        
        test1_passed = run_test_1()
        test2_passed = run_test_2()
        
        print("=" * 35)
        if test1_passed and test2_passed:
            print("✓ 모든 테스트 완료!")
            print("=" * 35)
            return 0
        else:
            print("✗ 일부 테스트 실패")
            print("=" * 35)
            return 1
    
    except Exception as e:
        print(f"오류 발생: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())


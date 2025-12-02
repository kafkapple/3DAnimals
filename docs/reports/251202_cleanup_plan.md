# 3DAnimals 프로젝트 정리 계획

**현재 총 용량**: 31GB
**목표**: 불필요한 파일 삭제로 용량 절감 및 프로젝트 구조 정리

---

## 용량 분석

| 폴더 | 용량 | 비고 |
|------|------|------|
| data/ | 21GB | horse.zip 14GB 포함 |
| results/ | 5.8GB | pretrained + 학습 결과 |
| outputs/ | 4.5GB | 대부분 삭제 가능 |
| model/ | 1.4MB | 유지 |
| logs/ | 1.3MB | 삭제 가능 |
| docs/ | 1.1MB | 유지 |
| config/ | 584KB | 정리 필요 (65개 파일) |

---

## 1단계: 즉시 삭제 가능 (예상 절감: ~18GB)

### 1.1 대용량 아카이브 파일 (14GB+)
```bash
# horse.zip - 다운로드 완료 후 압축 해제 안 함, 필요시 재다운로드
rm data/ponymation/horse.zip  # 14GB
```

### 1.2 outputs/ 폴더 전체 (4.5GB)
오래된 학습 결과, checkpoints, inference 결과 - 모두 삭제 가능
```bash
rm -rf outputs/  # 4.5GB
```

### 1.3 results/archive_251124/ (1.3GB)
이전 학습 아카이브 - 더 이상 불필요
```bash
rm -rf results/archive_251124/  # 1.3GB
```

### 1.4 pretrained zip 파일들 (이미 압축 해제됨)
```bash
# results/ponymation/ (~1.1GB)
rm results/ponymation/pretrained_*.zip

# results/magicpony/ (~700MB)
rm results/magicpony/pretrained_*.zip
```

---

## 2단계: Config 파일 정리 (65 → ~30개)

### 2.1 삭제할 copy/backup 파일
```bash
rm config/train_fauna\ copy.yaml
rm config/train_magicpony_horse\ copy.yaml
rm config/model/mouse_stable\ copy.yaml
```

### 2.2 구버전 mouse config 정리 (통합됨)
**삭제 대상** (train_fauna_mouse_*, finetune_* 로 대체됨):
```bash
# 구버전 train_mouse_*.yaml (13개)
rm config/train_mouse.yaml
rm config/train_mouse_debug.yaml
rm config/train_mouse_debug_stable.yaml
rm config/train_mouse_relaxed.yaml
rm config/train_mouse_relaxed_v2.yaml
rm config/train_mouse_relaxed_v3.yaml
rm config/train_mouse_scratch.yaml
rm config/train_mouse_stable.yaml
rm config/train_mouse_stage1.yaml
rm config/train_mouse_stage2.yaml
rm config/train_mouse_stage3.yaml
rm config/train_mouse_ultimate.yaml
rm config/train_mouse_ultimate_v3.yaml

# 구버전 model/mouse_*.yaml (10개)
rm config/model/mouse.yaml
rm config/model/mouse_relaxed.yaml
rm config/model/mouse_relaxed_v2.yaml
rm config/model/mouse_relaxed_v3.yaml
rm config/model/mouse_scratch.yaml
rm config/model/mouse_stable_.yaml
rm config/model/mouse_stage1.yaml
rm config/model/mouse_stage2.yaml
rm config/model/mouse_stage3.yaml
rm config/model/mouse_ultimate.yaml
rm config/model/mouse_ultimate_v3.yaml
```

---

## 3단계: 로그 및 임시 파일 정리

### 3.1 logs/ 폴더 (1.3MB)
```bash
rm -rf logs/  # 모든 구 로그 삭제
```

### 3.2 wandb 로컬 캐시
```bash
rm -rf wandb/  # 80KB, wandb 서버에 이미 저장됨
```

### 3.3 data 백업 폴더
```bash
rm -rf data/fauna/large_scale_backup/  # 존재시 삭제
```

---

## 4단계: results/ 정리 (선택적)

### 4.1 디버그 결과 (이미 테스트 완료)
```bash
# 디버그 결과 - 필요시 재생성 가능
rm -rf results/fauna_mouse_6view_debug/  # 291MB
```

### 4.2 실험 결과 (주의 필요)
```bash
# fauna/exp - 초기 실험, 불필요시 삭제
rm -rf results/fauna/exp/  # 존재시 확인 후 삭제
```

---

## 유지해야 할 파일

### Pretrained 모델 (압축 해제된 .pth)
```
results/magicpony/pretrained_horse/pretrained_horse.pth  ✓
results/ponymation/pretrained_horse/pretrained_horse_stage1.pth  ✓
results/ponymation/pretrained_horse/pretrained_horse_stage2.pth  ✓
results/fauna/pretrained_fauna/pretrained_fauna.pth  ✓
```

### 현재 사용 중인 Config
```
config/train_fauna_mouse_*.yaml  ✓
config/train_magicpony_mouse*.yaml  ✓
config/train_ponymation_mouse*.yaml  ✓
config/finetune_*.yaml  ✓
config/test_*.yaml  ✓
config/base*.yaml  ✓
config/model/fauna*.yaml  ✓
config/model/magicpony*.yaml  ✓
config/model/ponymation*.yaml  ✓
```

### Data
```
data/fauna_mouse/  ✓ (실제 학습 데이터)
data/magicpony/mouse/  ✓
data/ponymation/mouse/  ✓
data/tets/  ✓ (DMTet 필수)
```

---

## 실행 스크립트

### cleanup.sh (단계별 실행 권장)
```bash
#!/bin/bash
set -e

echo "=== Step 1: 대용량 파일 삭제 ==="
rm -f data/ponymation/horse.zip
rm -rf outputs/
rm -rf results/archive_251124/

echo "=== Step 2: zip 파일 삭제 (이미 압축해제됨) ==="
rm -f results/ponymation/pretrained_*.zip
rm -f results/magicpony/pretrained_*.zip

echo "=== Step 3: 구버전 config 삭제 ==="
rm -f config/*\ copy.yaml
rm -f config/model/*\ copy.yaml
rm -f config/train_mouse*.yaml
rm -f config/model/mouse_*.yaml
# 단, fauna_mouse, magicpony_mouse, ponymation_mouse는 유지

echo "=== Step 4: 로그/캐시 삭제 ==="
rm -rf logs/
rm -rf wandb/

echo "=== 완료 ==="
du -sh .
```

---

## 예상 결과

| 항목 | 삭제 용량 |
|------|----------|
| horse.zip | 14GB |
| outputs/ | 4.5GB |
| archive_251124/ | 1.3GB |
| pretrained zip들 | 1.8GB |
| 구버전 config | <1MB |
| logs, wandb | <2MB |
| **총 절감** | **~21GB** |

**최종 예상 용량**: 31GB → **~10GB**

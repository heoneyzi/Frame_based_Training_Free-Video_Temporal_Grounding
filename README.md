# Frame_based_Training_Free-Video_Temporal_Grounding
Frame-Level Understanding for  Lightweight and Explainable Video Temporal Grounding

# 📌 프레임 수준 유사도 기반의 경량형 Video Temporal Grounding 프레임워크 (FTF-VTG)

본 저장소는 **"Frame-Level Similarity for Lightweight and Explainable Video Temporal Grounding"** 논문의 공식 구현체입니다. 

제안한 모델은 **단 하나의 Vision-Language Model (VLM)**과 **추가 학습 없이**, **프레임-쿼리 간 유사도 곡선의 패턴만을 이용**하여 비디오 내 의미 있는 시간 구간을 탐지합니다. 

---

## 🌟 핵심 특징 (Highlights)

- **Training-Free**: 추가적인 학습 없이 사전학습된 VLM 하나만 사용
- **경량성**: GPU 메모리 **1GB 미만**으로 동작 가능
- **설명 가능성**: 유사도 곡선의 변화에 기반한 **직관적인 경계 탐지 방식**
- **경쟁력 있는 성능**: 기존 training-free 모델 대비 **최대 61% 성능 향상**
- **확장성**: **Video Moment Retrieval (VMR)** 과제에도 구조 변경 없이 적용 가능

---

## 📖 FTF-VTG의 실험 및 논문 개요

기존의 Video Temporal Grounding(VTG) 연구들은 대개 대규모 비디오 인코더, 복잡한 구조, Fine-tuning을 요구하는 모델들이 많았습니다. 이러한 방식은 높은 성능을 보장하지만, 실시간 처리나 저자원 환경에서는 적용에 한계가 있었습니다.

**본 연구는 정적인 프레임 시퀀스를 시간적으로 해석하는 접근을 통해**, 복잡한 구조 없이도 효과적으로 VTG 문제를 해결할 수 있음을 보여줍니다.

### ✅ 주요 구성 요소
- **프레임-쿼리 유사도 계산**: 사전학습된 VLM을 사용하여 각 프레임과 쿼리 간의 의미적 일치도를 계산
- **후처리 기반 유사도 곡선 분석**:
  - Gaussian smoothing으로 노이즈 제거
  - Morphological Gradient + 1차 미분 결합으로 유사도 변화 강조
  - Dual-threshold + Hysteresis로 경계 포착
- **후보 구간 병합 및 점수화**: 유사한 구간들을 병합하고, 구간 내/외부의 유사도 차이를 기반으로 점수화
- **최종 구간 선택**: 가장 신뢰도 높은 구간 하나를 예측 결과로 도출

### ✅ 실험 결과 요약
- **DiDeMo**와 **VidSTG** 두 가지 데이터셋에 대해 정량 평가
- **기존 training-free 방법 대비 성능 우위 확보**
- **추가 실험으로 VMR(검색) 과제로도 확장 가능함을 입증**
- **정성 평가를 통해 모델의 판단 근거를 시각적으로 확인 가능**

---

## 📁 데이터셋 다운로드

아래는 실험에 사용할 수 있도록 전처리된 JSON 파일입니다.

| Task 유형 | 데이터셋 | JSON 파일 링크 |
|-----------|-----------|----------------|
| VMR       | DiDeMo    | [vmr_DiDeMo_json](https://drive.google.com/file/d/1yalJNKS3c46drcDbSw5o5pgmmYJtPWQ6/view?usp=share_link) |
| VTG       | DiDeMo    | [vtg_DiDeMo_json](https://drive.google.com/file/d/1GE7DQLhf4Iw6SNFr-xnqjIamyg671s-1/view?usp=share_link) |
| VTG       | VidSTG    | [vtg_VidSTG_json](https://drive.google.com/file/d/1kHJLY3aDLRlRJrpBBOtRIDPfl2lZcSTQ/view?usp=share_link) |

> 다운로드 후 `FTF-VTG/data/extracted_json/` 디렉토리에 압축을 해제하세요.


---

## 🧪 실행 방법 (Usage)

FTF-VTG는 아래와 같이 간단한 명령어로 실행할 수 있습니다:

```bash
# VMR 실험 (DiDeMo)
python ftf_vtg/experiment/main.py --task vmr_didemo

# VTG 실험 (DiDeMo)
python ftf_vtg/experiment/main.py --task vtg_didemo

# VTG 실험 (VidSTG)
python ftf_vtg/experiment/main.py --task vtg_vidstg

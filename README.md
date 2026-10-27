# YOLO 실시간 작업 상태 이상 감지

웹캠에서 선택한 COCO 물체를 찾고, 물체 중심점이 화면 중앙의 작업 영역을 벗어나면 경고하는 프로그램입니다. 기본 감지 대상은 **cup (컵)**이며 실행 중에도 COCO 80개 라벨 중 하나를 선택할 수 있습니다.

## 주요 기능

- COCO 80개 라벨 선택 (영문/한글 병기)
- 가장 신뢰도가 높은 대상 하나를 추적
- 화면 중앙 작업 영역의 크기 조절
- 지정 프레임 연속 이탈 시 `WARNING`
- 대상 미감지 시 `OBJECT NOT DETECTED`
- 정상 영역으로 돌아오면 `RECOVERED` 후 `NORMAL`
- 신뢰도, 카메라 번호, 판정 프레임 수를 UI에서 조절

## 설치 및 실행

Python 3.10~3.12 환경을 권장합니다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

첫 실행에는 COCO로 사전 학습된 `yolo26n.pt` 모델이 자동 다운로드되므로 인터넷 연결이 필요합니다. 이후에는 다운로드된 모델을 재사용합니다. 창의 **카메라 시작** 버튼을 누르세요.

기본값은 컵, 신뢰도 0.4, 중앙 50%, 연속 5프레임입니다. `Q` 키로 종료할 수 있습니다.

```powershell
python main.py --target cup --camera 0 --confidence 0.4 --zone 0.5 --frames 5 --autostart
```

다른 Ultralytics 모델을 쓰려면 다음처럼 지정할 수 있습니다.

```powershell
python main.py --model yolo11n.pt
```

## 테스트

카메라 없이 상태 판정 로직을 확인할 수 있습니다.

```powershell
python -m unittest discover -s tests -v
```

## 문제 해결

- 카메라가 열리지 않으면 Zoom/Teams 등 카메라를 사용하는 프로그램을 종료하거나 카메라 번호를 1로 바꿔 보세요.
- Windows 설정에서 **개인 정보 및 보안 → 카메라 → 데스크톱 앱의 카메라 액세스**를 허용해야 합니다.
- GPU가 없어도 CPU로 실행되지만 FPS가 낮을 수 있습니다. 기본 `yolo26n.pt`는 nano 크기의 경량 모델입니다.

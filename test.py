import argparse
import os

# 각 태스크 함수 import
from ftf_vtg.experiment.vmr_DiDeMo import vmr_DiDeMo
from ftf_vtg.experiment.vtg_DiDeMo import vtg_DiDeMo
from ftf_vtg.experiment.vtg_VidSTG import vtg_VidSTG

# 기본 데이터 경로 설정
BASE_PATH = os.path.join("FTF-VTG", "data", "extracted_json")

# 경로 맵핑
TASK_JSON_DIRS = {
    "vmr_didemo": os.path.join(BASE_PATH, "vmr_DiDeMo_json"),
    "vtg_didemo": os.path.join(BASE_PATH, "vtg_DiDeMo_json"),
    "vtg_vidstg": os.path.join(BASE_PATH, "vtg_VidSTG_json"),
}

# 태스크와 함수 맵핑
TASK_FUNCTIONS = {
    "vmr_didemo": vmr_DiDeMo,
    "vtg_didemo": vtg_DiDeMo,
    "vtg_vidstg": vtg_VidSTG,
}


def main():
    parser = argparse.ArgumentParser(description="Run FTF-VTG experiments")
    parser.add_argument(
        "--task",
        choices=TASK_FUNCTIONS.keys(),
        required=True,
        help="Choose the task to run (e.g., vmr_didemo, vtg_didemo, vtg_vidstg)",
    )
    args = parser.parse_args()

    task = args.task
    json_dir = TASK_JSON_DIRS[task]
    task_func = TASK_FUNCTIONS[task]

    print(f"🔍 Running task: {task}")
    print(f"📂 Using data from: {json_dir}")
    task_func(json_dir)


if __name__ == "__main__":
    main()
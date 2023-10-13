import os
import shutil

# 폴더 A 에 있는 파일명을 C에 복사 하고 B에서는 해당 파일을 모두 삭제합니다. 이 스크립트는 성공/실패 파일을 분리하는데 쓰입니다.
# 폴더 A, B, C 의 이름을 여기에 입력하세요.
FOLDER_A = 'C:\\Users\\enmus\\FindHolyGrail\\Target0002\\holygrail'
FOLDER_B = 'C:\\Users\\enmus\\FindHolyGrail\\Target0002\\PickByAi'
FOLDER_C = 'C:\\Users\\enmus\\FindHolyGrail\\Target0002\\success'

def move_files_from_folder_B_to_C_based_on_folder_A(folder_a, folder_b, folder_c):
    # 폴더 A에서 파일 이름들을 추출
    files_in_A = [f for f in os.listdir(folder_a) if os.path.isfile(os.path.join(folder_a, f))]
    base_names_in_A = [os.path.splitext(f)[0] for f in files_in_A]  # 확장자 제거

    # 폴더 C가 없다면 생성
    if not os.path.exists(folder_c):
        os.makedirs(folder_c)

    # 폴더 B의 모든 파일을 순회하며 검사
    for root, _, files in os.walk(folder_b):
        for file_in_B in files:
            # 확장자를 제거한 파일명이 폴더 A의 파일 이름을 포함하는 경우, 옮기기
            base_name_in_B = os.path.splitext(file_in_B)[0]
            for base_name_in_A in base_names_in_A:
                if base_name_in_A in base_name_in_B:
                    file_to_move = os.path.join(root, file_in_B)
                    destination = os.path.join(folder_c, file_in_B)
                    try:
                        shutil.move(file_to_move, destination)
                        print(f"Moved: {file_to_move} to {destination}")
                    except Exception as e:
                        print(f"Error moving {file_to_move}: {e}")

if __name__ == "__main__":
    move_files_from_folder_B_to_C_based_on_folder_A(FOLDER_A, FOLDER_B, FOLDER_C)

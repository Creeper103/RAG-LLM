import os

def read_files_in_folder(folder_path):
    contents = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    contents.append(f.read())
            except Exception as e:
                print(f"讀取檔案失敗: {file_path}，錯誤: {e}")
    return contents

def merge_data(language_folder):
    base_path = os.path.abspath("data")
    merge_folders = ["power", "sensors", "structure"]
    all_contents = []

    for folder in merge_folders:
        folder_path = os.path.join(base_path, language_folder, folder)
        if os.path.exists(folder_path):
            print(f"處理資料夾: {folder_path}")
            all_contents.extend(read_files_in_folder(folder_path))
        else:
            print(f"資料夾不存在: {folder_path}")

    merged_text = "\n\n".join(all_contents)
    output_file = os.path.join(base_path, f"merged_{language_folder}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(merged_text)
    print(f"合併後檔案已儲存至: {output_file}")

if __name__ == "__main__":
    # 你想合併哪個語言的資料夾？"en" 或 "ch"
    merge_data("en")
    merge_data("ch")

import pandas as pd

def preview_csv(file_path, encoding='utf-8'):
    
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print("前 5 行数据:")
        print(df.head())
        print("======")
        print(df.iloc[0:5,0:10])
    except FileNotFoundError:
        print("文件未找到，请检查路径是否正确。")
    except Exception as e:
        print("读取文件时出错:", e)

if __name__ == "__main__":
    path = "files\data_files\mnist\mnist_train.csv"
    preview_csv(path)

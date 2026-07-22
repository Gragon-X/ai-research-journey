def load_grades(filename):
    scores = []#存储数据
    try:
        with open(filename, 'r') as f:
                for line_num, line in enumerate(f, start=1):#寻找异常
                    try:
                        num = int(line.strip())
                        scores.append(num)
                    except ValueError:
                        print(f"第{line_num}行格式错误：'{line.strip()}' 无法转化数字")
    except FileNotFoundError:
        print("文件未找到。")
        with open(filename, 'w') as f:
            pass  # 创建一个空文件
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
    return scores
result = load_grades("grade.txt")
print(f"成功读取的成绩：{result}")

#定义函数创建文件
def students_score(students,filename="students_score.txt"):
    with open(filename,"w") as f:
        for student in students:
            f.write(f"{student['name']}的成绩是{student['score']}\n")
        print(f"学生成绩已写入文件 {filename} 中。")
#date
students = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name":"carol","score":78}
]
students_score(students, "students_score.txt")
with open("students_score.txt", "r") as f:
    content = f.read()
print("文件内容如下：")
print(content)
import os
print(os.path.abspath("students.txt"))
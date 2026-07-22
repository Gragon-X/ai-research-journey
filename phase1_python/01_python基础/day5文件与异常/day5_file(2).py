def student_score(name, score):
    with open("students_score.txt", "a") as f:
        f.write(f"{name}的成绩是：{score}\n")
    print(f"{name}的成绩已写入文件 students_score.txt 中。")
while True:
    name = input("请输入学生姓名：")
    score = input("请输入学生成绩：")
    student_score(name, score)
    choice = input("是否继续？(y/n)：")
    if choice == "n":
        break
with open("students_score.txt", "r") as f:
    print("\n=== 所有学生成绩 ===")
    print(f.read())
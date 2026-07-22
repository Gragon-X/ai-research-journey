#定义类方便使用
class Students:
    def __init__(self,name,score):
        self.name = name
        self.score = score
    def __str__(self):
        return f"{self.name}:{self.score}分"
#初始加载数据
def load_from_file (filename = "students.txt"):
    students_list = []
    try:
        with open(filename,"r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    name,score_str = line.split(",")
                    score = int(score_str)
                    students_list.append(Students(name,score))
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    return students_list
#更新数据
def update_score(students_list):
    name = input("请输入需要修改的姓名：")
    for s in students_list:
        if s.name == name:
            try:
                new_score = int(input("请输入新的成绩："))
                s.score = new_score
                print("成绩修改成功！")
            except ValueError:
                print("成绩应为数字！")
            return
    print("查无此人！")
#查询所有数据
def check(students_list):
    if not students_list:
        print("暂无学生数据！")
        return
    for s in students_list:
        print(f"{s.name}:{s.score}分")
#添加数据
def add(students_list):
    while True:
        name = input("姓名：")
        try:
            score = int(input("成绩："))
        except ValueError:
            print("成绩必须是数字！")
            continue
        students_list.append(Students(name,score))
        choice = input("是否继续？(y/n):")
        if choice == "n" or choice == "N":
            break
#保存数据
def save(students_list):
    with open("students.txt","w") as f:
        for s in students_list:
            f.write(f"{s.name},{s.score}\n")
#删除数据
def delete_student(students_list):
    name = input("请输入要删除的姓名：")
    for i,s in enumerate(students_list):
        if s.name == name:
            del students_list[i]
            print("删除成功！")
            return
    print("查无此人！")
#统计数据
def show_state(students_list):
    if not students_list:
        print("暂无学生数据！")
        return
    score = [s.score for s in students_list]
    print(f"参考总人数为：{len(students_list)}")
    print(f"平均分为：{sum(score)/len(students_list):.2f}")
    print(f"最高分为：{max(score)}")
    print(f"最低分为：{min(score)}")
#主程序
def main():
    students_list = load_from_file()
    while True:
        print("======教务系统======")
        print("1.查看")
        print("2.添加")
        print("3.修改")
        print("4.删除")
        print("5.统计")
        print("0.退出")
        print("====================")
        choice = input("选择：")
        if choice == "1":
            check(students_list)
            save(students_list)
        elif choice == "2":
            add(students_list)
            save(students_list)
        elif choice == "3":
            update_score(students_list)
            save(students_list)
        elif choice == "4":
            delete_student(students_list)
            save(students_list)
        elif choice == "5":
            show_state(students_list)
            save(students_list)
        elif choice == "0":
            print("再见！")
            save(students_list)
            break
        else:
            print("请输入上述有意义的数字！")
        print("请继续：")
if __name__ == "__main__":
    main()
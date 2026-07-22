class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score
def load_students(filename):
    students=[]
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            line = line.split(",")
            name = line[0]
            score = int(line[1])
            s = Student(name, score)
            students.append(s)
    return students
result = load_students("students.txt")
for s in result:
    print(s.name, s.score)
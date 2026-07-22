students = {
"2021001": {"name": "Alice", "score": 85},
"2021002": {"name": "Bob", "score": 92},
"2021003": {"name": "Charlie", "score": 78}
}
index = input("Enter student ID: ")
if index in students:
    student = students[index]
    print(f"Name: {student['name']}, Score: {student['score']}")
else:
    print("Student ID not found.")
total_score = 0
for student in students.values():
    total_score += student["score"]
    average_score = total_score / len(students)
print(f"Average score: {average_score}")
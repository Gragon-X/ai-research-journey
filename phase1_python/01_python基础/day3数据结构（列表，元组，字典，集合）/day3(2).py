scores = [78, 85, 92, 67, 88, 79, 95, 81]

max_score = 0      # 改名字
min_score = 100
total = 0
count_90 = 0

for score in scores:
    total += score
    if score > max_score:
        max_score = score
    if score < min_score:
        min_score = score
    if score >= 90:
        count_90 += 1

print(f"最高分: {max_score}")
print(f"最低分: {min_score}")
print(f"平均分: {total / len(scores):.2f}")
print(f"从高到低: {sorted(scores, reverse=True)}")
print(f"90分以上的人数: {count_90}")
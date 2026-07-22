def analyze_score(score):
    max_score = 0      
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

    return {
        "max": max_score,
        "min": min_score,
        "avg": round(total / len(scores), 2),
        "count_90": count_90,
        "sorted": sorted(scores, reverse=True)
    }
scores = [78, 85, 92, 67, 88, 79, 95, 81]
result = analyze_score(scores)

print(f"最高分: {result['max']}")
print(f"最低分: {result['min']}")
print(f"平均分: {result['avg']}")
print(f"90分以上: {result['count_90']}人")
print(f"排名: {result['sorted']}")
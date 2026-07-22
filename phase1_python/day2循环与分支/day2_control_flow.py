answer = 30
a = int(input("Guess a number between 0 and 1000: "))
while a != answer:
    if a > answer:
        print("Too high")
        a=int(input("Guess again: "))
    elif a < answer:
        print("Too low")
        a=int(input("Guess again: "))
if a == answer:
    print("Correct!")
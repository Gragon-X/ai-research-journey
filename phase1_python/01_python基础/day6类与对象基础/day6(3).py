class BankAccount:
    def __init__(self,owener,balance):
        self.owener = owener
        self.balance = balance
    def deposit(self,amount):
        self.balance +=amount
    def withdraw(self,amount):
        if amount > self.balance:
            print("余额不足！")
        else:
            self.balance -= amount
    def get_balance(self):
        return self.balance
acc = BankAccount("Alice", 100)
acc.deposit(50)      # 余额：150
acc.withdraw(200)    # 余额不足，取款失败
acc.withdraw(30)     # 余额：120
print(acc.get_balance())  # 120
class Book:
    def __init__(self,title,author,price):
        self.title = title
        self.author = author
        self.price = price
    def print_info(self):
        print(f"《{self.title}》作者：{self.author}，价格：{self.price}")
    def is_affordable(self,budget):
        if self.price <= budget:
            return True
        else:
            return False
b1 = Book("Python编程", "张三", 59.9)
b2 = Book("深度学习", "李四", 128.0)
b1.print_info()
print(b1.is_affordable(100))
print(b2.is_affordable(100))
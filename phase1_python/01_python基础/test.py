import sys
print("Python路径:", sys.executable)
print("版本:", sys.version)
#测试时conda是否正常
try:
    import numpy as np
    print("numpy版本:", np.__version__)
    print("numpy导入成功")
except ImportError:
    print("numpy导入失败,请检查conda环境是否正常")
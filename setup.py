from setuptools import setup, find_packages

# 初始化设置
setup(
    name="medical_chatbot",
    version="0.1.0",
    author="Boktiar Ahmed Bappy",
    author_email="entbappy73@gmail.com",
    # 自动发现所有包
    packages=find_packages(),
    # 依赖项列表，会自动从 requirements.txt 中读取，然后尝试安装
    install_requires=[]
)
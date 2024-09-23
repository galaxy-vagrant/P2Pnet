# 计算txt文件中的行数
def count_lines(filename):
    """
    计算并返回指定文件中的行数。
    
    :param filename: 文件的路径
    :return: 文件中的行数
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return sum(1 for line in file)
    except FileNotFoundError:
        print(f"文件 {filename} 未找到。")
        return None
    except Exception as e:
        print(f"读取文件 {filename} 时发生错误：{e}")
        return None

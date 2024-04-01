
def load(cell_line):
    x_region = 'data/' + cell_line + '/' + 'x.fasta'
    y_region = 'data/' + cell_line + '/' + 'y.fasta'
    x_sequence = []
    y_sequence = []

    f = open(x_region, 'r')
    for i in f.readlines():
        if i[0] != ' ':
            if i[0] != '>':
                x_sequence.append(i.strip().upper())
    f.close()

    f = open(y_region, 'r')
    for i in f.readlines():
        if i[0] != ' ':
            if i[0] != '>':
                y_sequence.append(i.strip().upper())
    f.close()
    return x_sequence, y_sequence


def load_Bi(cell_line):
    """
    从两个FASTA文件中读取DAN序列数据, 并转换为列表形式

    :param cell_line: 表示要处理的细胞系
    :return: 返回正向和反向序列 列表 (是一个一维数组, 存的是string类型)
    """
    x_region = 'data/' + cell_line + '/' + 'x.fasta'
    y_region = 'data/' + cell_line + '/' + 'y.fasta'
    x_forward = []
    x_reverse = []
    y_forward = []
    y_reverse = []

    # 打开文件, 按行读取内容, 将序列数据存入正向和反向(将序列反转)列表中
    f = open(x_region, 'r')
    for i in f.readlines():
        if i[0] != ' ':
            if i[0] != '>':
                x_forward.append(i.strip().upper())
                x_reverse.append(i.strip().upper()[::-1])  # 移除字符串i两端的空白, 大写, 再反转
    f.close()

    f = open(y_region, 'r')
    for i in f.readlines():
        if i[0] != ' ':
            if i[0] != '>':
                y_forward.append(i.strip().upper())
                y_reverse.append(i.strip().upper()[::-1])
    f.close()
    return x_forward, x_reverse, y_forward, y_reverse

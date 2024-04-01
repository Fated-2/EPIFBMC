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
    x_region = 'data/' + cell_line + '/' + 'x.fasta'
    y_region = 'data/' + cell_line + '/' + 'y.fasta'
    x_forward = []
    x_reverse = []
    y_forward = []
    y_reverse = []

    f = open(x_region, 'r')
    for i in f.readlines():
        if i[0] != ' ':
            if i[0] != '>':
                x_forward.append(i.strip().upper())
                x_reverse.append(i.strip().upper()[::-1])
    f.close()

    f = open(y_region, 'r')
    for i in f.readlines():
        if i[0] != ' ':
            if i[0] != '>':
                y_forward.append(i.strip().upper())
                y_reverse.append(i.strip().upper()[::-1])
    f.close()
    return x_forward, x_reverse, y_forward, y_reverse

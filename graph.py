import matplotlib.pyplot as plt
from sys import argv, exit
import json


def graph(fname, column):
    with open(fname, 'r') as inf:
        data = inf.read()
        lines = data.split('\n')
        info = lines[0]
        jsinfo = None
        if '}' in info:
            jsinfo = info[:info.index('}') + 1]
            print(jsinfo)
            start_line = info[info.index('}') + 1:]
            if start_line:
                lines = [start_line.split(',')] + [l.split(',') for l in lines[1:-1]]
            else:
                lines = [l.split(',') for l in lines[1:-1]]
            info = json.loads(jsinfo)
        else:
            lines = [l.split(',') for l in lines[:-1]]

    print(lines)
    if column == 'time':
        plt.plot([l[2] for l in lines], [float(l[1]) for l in lines])
    elif column == 'iter':
        plt.plot([l[0] for l in lines], [float(l[1]) for l in lines])
    else:
        print('Bad input')
        exit(1)

    if jsinfo:
        title = '{} {} graphed by {}'.format('mnist' if 'mnist' in fname else 'Numbers',
                                             ','.join(['{}={}'.format(k, info[k]) for k in info.keys()]), column)
    else:
        title = fname
    plt.suptitle(title)
    plt.show()


if __name__ == '__main__':
    if len(argv) < 2:
        from glob import glob
        fname = glob('logs/num*.log')[-1]
        column = 'iter'
    else:
        fname = argv[1]
        column = argv[2]
    graph(fname, column)

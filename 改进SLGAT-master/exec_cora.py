# coding=utf-8

import os


def generate_command(opt):
    cmd = 'python -u train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd


def run(opt):
    os.system(generate_command(opt))


def default_opt():
    opt = {}
    opt['model'] = 'SLGAT'
    opt['lr'] = 0.05
    opt['hidden'] = 32
    opt['class-hidden'] = 16
    opt['dropout'] = 0.5
    opt['input-dropout'] = 0.5
    opt['weight-dropout'] = 0.5
    opt['pre-epochs'] = 200
    opt['epochs'] = 600

    opt['data'] = 'cora'
    opt['file'] = 'cora.res.txt'
    return opt


if __name__ == '__main__':
    opt = default_opt()
    print(opt, flush=True)
    with open(opt['file'], 'w') as f:
        pass
    for k in range(100):
        seed = k + 1
        opt['seed'] = seed
        run(opt)

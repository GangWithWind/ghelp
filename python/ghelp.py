import os

def update(_):
    import git
    folder = os.path.dirname(__file__)
    repo = git.Repo(os.path.dirname(folder))
    remote = repo.remote()
    info = remote.pull()
    for item in info:
        print(item)

def bunch(args):
    import glob
    folder = args.folder
    filter = args.filter
    subfolders = glob.glob(f'{folder}/{filter}/param.dat')
    cmd0 = args.cmd_str

    with open(f'{folder}/{args.project}.ps1', 'w') as fid:
        for subfolder in subfolders:
            subfolder = os.path.dirname(subfolder)
            cmd = cmd0.replace('%TARGET', subfolder)
            print(cmd)
            fid.write(cmd + '\n')

def show_list(_):
    print("Wellcome to ghelp, here's a list of functions.")
    print("-----------")
    print('ghelp\tSome small tools.')
    print('g2fits\tTransfer output of genga to fits file.')
    print('gdisk\tMake genga configration files for disk.')
    print('gvideo\tMake video to demonstrate the output.')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='tool name')
    parser_update = subparsers.add_parser('list', help='list all the functions of gtools')
    parser_update.set_defaults(func=show_list)

    parser_update = subparsers.add_parser('update', help='self update of ghelp tools')
    parser_update.set_defaults(func=update)

    parser_bunch = subparsers.add_parser('bunch', help='self update of ghelp tools')
    parser_bunch.add_argument('folder', type=str, help='base folder to process')
    parser_bunch.add_argument('-p', '--project', type=str, help='output scriptname', default='bunch')
    parser_bunch.add_argument('-f', '--filter', type=str, help='dir filter', default='*')
    parser_bunch.add_argument('cmd_str', type=str, help='argument string')
    parser_bunch.set_defaults(func=bunch)

    args = parser.parse_args()
    args.func(args)



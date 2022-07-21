import os


if __name__ == '__main__':
    dir = r'F:\lhc\ABAW4\MTL\dataset\test_aligned'
    txtpath = r'./test.txt'
    with open(txtpath, 'w') as f:
        f.write('image' + '\n')
        for case in os.listdir(dir):
            case_path = os.path.join(dir, case)
            for file in os.listdir(case_path):
                f.write(case + '/' + file + '\n')
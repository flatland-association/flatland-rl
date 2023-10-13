
from flatland.utils import env_edit_utils as eeu


def test_makeTestEnv():
    eeu.makeTestEnv()

    for sNameEnv in eeu.ddEnvSpecs.keys():
        print(sNameEnv)
        eeu.makeTestEnv(sNameEnv)



def main():
    test_makeTestEnv()

if __name__ == '__main__':
    main()


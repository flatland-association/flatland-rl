
from flatland.utils import env_edit_utils as eeu


def test_makeTestEnv():
    eeu.makeTestEnv(nAg=0)

    for sNameEnv in eeu.ddEnvSpecs.keys():
        print(sNameEnv)
        eeu.makeTestEnv(sNameEnv, nAg=0)



def main():
    test_makeTestEnv()

if __name__ == '__main__':
    main()


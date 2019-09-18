#!/usr/bin/env python
import glob
import os
import shutil
import subprocess
import webbrowser
from urllib.request import pathname2url


def browser(pathname):
    webbrowser.open("file:" + pathname2url(os.path.abspath(pathname)))


def remove_exists(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


# clean docs config and html files, and rebuild everything
# wildcards do not work under Windows
for file in glob.glob(r'./docs/flatland*.rst'):
    remove_exists(file)
remove_exists('docs/modules.rst')

# copy md files from root folder into docs folder
for file in glob.glob(r'./*.md'):
    print(file)
    shutil.copy(file, 'docs/')

subprocess.call(['sphinx-apidoc', '--force', '-a', '-e', '-o', 'docs/', 'flatland', '-H', '"Flatland Reference"'])

os.environ["SPHINXPROJ"] = "flatland"
os.chdir('docs')
subprocess.call(['python', '-msphinx', '-M', 'clean', '.', '_build'])
# TODO fix sphinx warnings instead of suppressing them...
subprocess.call(['python', '-msphinx', '-M', 'html', '.', '_build'])
#subprocess.call(['python', '-msphinx', '-M', 'html', '.', '_build', '-Q'])

# we do not currrently use pydeps, commented out https://gitlab.aicrowd.com/flatland/flatland/issues/149
# subprocess.call(['python', '-mpydeps', '../flatland', '-o', '_build/html/flatland.svg', '--no-config', '--noshow'])

browser('_build/html/index.html')

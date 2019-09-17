#!/usr/bin/env python
import glob
import os
import shutil
import subprocess
import webbrowser
from urllib.request import pathname2url
from shutil import copyfile

def browser(pathname):
    webbrowser.open("file:" + pathname2url(os.path.abspath(pathname)))


def remove_exists(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


# clean docs config and html files, and rebuild everything
remove_exists('docs/flatland*.rst')
remove_exists('docs/modules.rst')

subprocess.call(['sphinx-apidoc', '--force', '-a', '-e', '-o', 'docs/', 'flatland', '-H', '"Flatland Reference"'])

for file in glob.glob(r'./*.md'):
    print(file)
    shutil.copy(file, 'docs/')

os.environ["SPHINXPROJ"] = "flatland"
os.chdir('docs')
subprocess.call(['python', '-msphinx', '-M', 'clean', '.', '_build'])
# TODO fix sphinx warnings instead of suppressing them...
subprocess.call(['python', '-msphinx', '-M', 'html', '.', '_build', '-Q'])
subprocess.call(['python', '-mpydeps', '../flatland', '-o', '_build/html/flatland.svg', '--no-config', '--noshow'])

browser('_build/html/index.html')

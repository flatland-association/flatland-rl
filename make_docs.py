#!/usr/bin/env python

import os
import webbrowser
import subprocess
from urllib.request import pathname2url

def browser(pathname):
    webbrowser.open("file:" + pathname2url(os.path.abspath(pathname)))

def remove_exists(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


# clean docs config and html files, and rebuild everything
remove_exists('docs/flatland.rst')
remove_exists('docs/modules.rst')

subprocess.call(['sphinx-apidoc', '-o', 'docs/', 'flatland'])

os.environ["SPHINXPROJ"] = "flatland"
os.chdir('docs')
subprocess.call(['python', '-msphinx', '-M', 'clean', '.', '_build'])
subprocess.call(['python', '-msphinx', '-M', 'html', '.', '_build'])

browser('_build/html/index.html')

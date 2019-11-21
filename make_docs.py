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
for image_file in glob.glob(r'./docs/flatland*.rst'):
    remove_exists(image_file)
remove_exists('docs/modules.rst')

for md_file in glob.glob(r'./*.md') + glob.glob(r'./docs/specifications/*.md') + glob.glob(r'./docs/tutorials/*.md'):
    from m2r import parse_from_file

    rst_content = parse_from_file(md_file)
    rst_file = md_file.replace(".md", ".rst")
    remove_exists(rst_file)
    with open(rst_file, 'w') as out:
        print("m2r {}->{}".format(md_file, rst_file))

        out.write(rst_content)
        out.flush()

img_dest = 'docs/images/'
if not os.path.exists(img_dest):
    os.makedirs(img_dest)
for image_file in glob.glob(r'./images/*.png'):
    shutil.copy(image_file, img_dest)

subprocess.call(['sphinx-apidoc', '--force', '-a', '-e', '-o', 'docs/', 'flatland', '-H', 'API Reference', '--tocfile',
                 '05_apidoc'])

os.environ["SPHINXPROJ"] = "Flatland"
os.chdir('docs')
subprocess.call(['python', '-msphinx', '-M', 'clean', '.', '_build'])
img_dest = '_build/html/img'
if not os.path.exists(img_dest):
    os.makedirs(img_dest)
for image_file in glob.glob(r'./specifications/img/*'):
    shutil.copy(image_file, img_dest)
subprocess.call(['python', '-msphinx', '-M', 'html', '.', '_build'])

# we do not currrently use pydeps, commented out https://gitlab.aicrowd.com/flatland/flatland/issues/149
# subprocess.call(['python', '-mpydeps', '../flatland', '-o', '_build/html/flatland.svg', '--no-config', '--noshow'])

browser('_build/html/index.html')

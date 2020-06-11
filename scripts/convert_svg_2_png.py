#!/usr/bin/env python

import glob
from cairosvg import svg2png
import os
import shutil
import tqdm

import io
from PIL import Image

########################################################
########################################################
# 
# Converts SVG assets into PNG assets
# 
# We use this approach to drop the CairoSVG dependency
# from the flatland requirements.
# 
# Usage Requires :
#
# conda install cairo
#
########################################################
########################################################

TARGET_PNG_WIDTH=300
TARGET_PNG_HEIGHT=300

SVG_FOLDER="../flatland/svg"
TARGET_FOLDER="../flatland/png"

# Delete target PNG files, if they exist
for _png_file in glob.glob(os.path.join(TARGET_FOLDER, "*.png")):
    os.remove(_png_file)

# Convert all SVG files into PNG files
for _source_svg_path in tqdm.tqdm(glob.glob(os.path.join(SVG_FOLDER, "*.svg"))):
    base_filename = os.path.basename(_source_svg_path)
    target_filename = base_filename.replace(".svg", ".png")
    target_filepath = os.path.join(
        TARGET_FOLDER,
        target_filename
    )
    bytesPNG = svg2png(
                    file_obj=open(_source_svg_path, "rb"),
                    output_height=TARGET_PNG_WIDTH,
                    output_width=TARGET_PNG_HEIGHT
                )
    with io.BytesIO(bytesPNG) as fIn:
        im = Image.open(fIn)
        im.load()
        assert im.size == (TARGET_PNG_WIDTH, TARGET_PNG_HEIGHT)
        im.save(target_filepath)


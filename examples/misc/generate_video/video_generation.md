# Making Videos from Env

In order to generate Videos or gifs, it is easiest to generate image files and then run ffmpeg to generate a video.

## 1. Generating Images from Env

Start by importing the render and instantiating it

```
from flatland.utils.rendertools import RenderTool
env_renderer = RenderTool(env, gl="PILSVG", )
```

If the environment changes don't forget to reset the renderer
```
env_renderer.reset()
```

You can now record an image after every step. It is best to use a format similar to the one below, where `frame_step` is counting the number of steps.
```
env_renderer.gl.save_image("./Images/Avoiding/flatland_frame_{:04d}.bmp".format(frame_step))
```

Once the images have been saved to the folder you can run a shell from that folder and run the following commands.

Generate a mp4 out of the images:
```
ffmpeg -y -framerate 12 -i flatland_frame_%04d.bmp -hide_banner -c:v libx264 -pix_fmt yuv420p test.mp4
```

Generate a palette out of the video necessary to generate beautiful gifs:
```
ffmpeg  -i test.mp4 -filter_complex "[0:v] palettegen" palette.png
```
and finaly generate the gif
```
ffmpeg -i test.mp4 -i palette.png -filter_complex "[0:v][1:v] paletteuse" single_agent_navigation.gif
```

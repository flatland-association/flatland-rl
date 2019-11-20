# FAQ about bugs

### Why are my trains drawn outside of the rails?
If you render your environment and the agents appear to be off the rail it is usually due to changes in the railway infrastructure. Make sure that you reset your renderer anytime the infrastructure changes by calling `env_renderer.reset().
`
### I keep getting there error when submitting from windows
When submitting from a Windows system you might run into the following error:
```
OSError: dlopen() failed to load a library: cairo / cairo-2 / cairo-gobject-2 / cairo.so.2
```

Please follow the intstruction in the starter-kit to avoid these problems. Make sure to reset your `environment.yml` correctly.

[Link to Windows submission instructions](https://github.com/AIcrowd/flatland-challenge-starter-kit/blob/master/windows_submission.md)

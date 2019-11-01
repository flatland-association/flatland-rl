# FAQ about bugs

### Why are my trains drawn outside of the rails?
If you render your environment and the agents appear to be off the rail it is usually due to changes in the railway infrastructure. Make sure that you reset your renderer anytime the infrastructure changes by calling `env_renderer.reset().
`

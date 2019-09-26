## Visualization

![logo](https://drive.google.com/uc?export=view&id=1rstqMPJXFJd9iD46z1A5Rus-W0Ww6O8i)


### Introduction & Scope

Broad requirements for human-viewable display of a single Flatland Environment.


#### Context

Shows this software component in relation to some of the other components.  We name the component the "Renderer".  Multiple agents interact with a single Environment.  A renderer interacts with the environment, and displays on screen, and/or into movie or image files.



<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline drawings not supported directly from Docs. You may want to copy the inline drawing to a standalone drawing and export by reference. See <a href="https://github.com/evbacher/gd2md-html/wiki/Google-Drawings-by-reference">Google Drawings by reference</a> for details. The img URL below is a placeholder. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![drawing](https://docs.google.com/a/google.com/drawings/d/12345/export/png)


### Requirements


#### Primary Requirements



1. Visualize or Render the state of the environment
    1. Read an Environment + Agent Snapshot provided by the Environment component
    2. Display onto a local screen in real-time (or near real-time)
    3. Include all the agents
    4. Illustrate the agent observations (typically subsets of the grid / world)
    5. 2d-rendering only
2. Output visualisation into movie / image files for use in later animation
3. Should not impose control-flow constraints on Environment
    6. Should not force env to respond to events
    7. Should not drive the "main loop" of Inference or training 


#### Secondary / Optional Requirements 



1. During training (possibly across multiple processes or machines / OS instances), display a single training environment,
    1. without holding up the other environments in the training.
    2. Some training environments may be remote to the display machine (eg using GCP / AWS)
    3. Attach to / detach from running environment / training cluster without restarting training.
2. Provide a switch to make use of graphics / artwork provided by graphic artist
    4. Fast / compact mode for general use
    5. Beauty mode for publicity / demonstrations
3. Provide a switch between smooth / continuous animation of an agent (slower) vs jumping from cell to cell (faster)
    6. Smooth / continuous translation between cells
    7. Smooth / continuous rotation 
4. Speed - ideally capable of 60fps (see performance metrics)
5. Window view - only render part of the environment, or a single agent and agents nearby.
    8. May not be feasible to render very large environments
    9. Possibly more than one window, ie one for each selected agent
    10. Window(s) can be tied to agents, ie they move around with the agent, and optionally rotate with the agent.
6. Interactive scaling
    11. eg wide view, narrow / enlarged view
    12. eg with mouse scrolling & zooming
7. Minimize necessary skill-set for participants
    13. Python API to gui toolkit, no need for C/C++
8. View on various media:
    14. Linux & Windows local display
    15. Browser


#### Performance Metrics

Here are some performance metrics which the Renderer should meet.


<table>
  <tr>
   <td>
   </td>
   <td><p style="text-align: right">
### Per second</p>

   </td>
   <td><p style="text-align: right">
Target Time (ms)</p>

   </td>
   <td><p style="text-align: right">
Prototype time (ms)</p>

   </td>
  </tr>
  <tr>
   <td>Write an agent update (ie env as client providing an agent update)
   </td>
   <td>
   </td>
   <td><p style="text-align: right">
0.1</p>

   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Draw an environment window 20x20
   </td>
   <td><p style="text-align: right">
60</p>

   </td>
   <td><p style="text-align: right">
16</p>

   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Draw an environment window 50 x 50
   </td>
   <td><p style="text-align: right">
10</p>

   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Draw an agent update on an existing environment window.  5 agents visible.
   </td>
   <td>
   </td>
   <td><p style="text-align: right">
1</p>

   </td>
   <td>
   </td>
  </tr>
</table>



#### Example Visualization


### Reference Documents

Link to this doc: https://docs.google.com/document/d/1Y4Mw0Q6r8PEOvuOZMbxQX-pV2QKDuwbZJBvn18mo9UU/edit#


#### Core Specification

This specifies the system containing the environment and agents - this will be able to run independently of the renderer.

[https://docs.google.com/document/d/1RN162b8wSfYTBblrdE6-Wi_zSgQTvVm6ZYghWWKn5t8/edit](https://docs.google.com/document/d/1RN162b8wSfYTBblrdE6-Wi_zSgQTvVm6ZYghWWKn5t8/edit)

The data structure which the renderer needs to read initially resides here.


#### Visualization Specification

This will specify the software which will meet the requirements documented here.

[https://docs.google.com/document/d/1XYOe_aUIpl1h_RdHnreACvevwNHAZWT0XHDL0HsfzRY/edit#](https://docs.google.com/document/d/1XYOe_aUIpl1h_RdHnreACvevwNHAZWT0XHDL0HsfzRY/edit#)


#### Interface Specification

This will specify the interfaces through which the different components communicate


### Non-requirements - to be deleted below here.

The below has been copied into the spec doc.    Comments may be lost.  I'm only preserving it to save the comments for a few days - they don't cut & paste into the other doc!


#### Interface with Environment Component



*   Environment produces the Env Snapshot data structure (TBD)
*   Renderer reads the Env Snapshot
*   Connection between Env and Renderer, either:
    *   Environment "invokes" the renderer in-process
    *   Renderer "connects" to the environment
        *   Eg Env acts as a server, Renderer as a client
*   Either
    *   The Env sends a Snapshot to the renderer and waits for rendering
*   Or:
    *   The Env puts snapshots into a rendering queue
    *   The renderer blocks / waits on the queue, waiting for a new snapshot to arrive
        *   If several snapshots are waiting, delete and skip them and just render the most recent
        *   Delete the snapshot after rendering
*   Optionally
    *   Render every frame / time step
    *   Or, render frames without blocking environment
        *   Render frames in separate process / thread


###### Environment Snapshot

**Data Structure**

A definitions of the data structure is to be defined in Core requirements.

It is a requirement of the Renderer component that it can read this data structure.

**Example only**

Top-level dictionary



*   World nd-array
    *   Each element represents available transitions in a cell
*   List of agents
    *   Agent location, orientation, movement (forward / stop / turn?)
    *   Observation
        *   Rectangular observation
            *   Maybe just dimensions - width + height (ie no need for contents)
            *   Can be highlighted in display as per minigrid
        *   Tree-based observation
            *   TBD


#### Investigation into Existing Tools / Libraries



1. Pygame
    1. Very easy to use. Like dead simple to add sprites etc. ([https://studywolf.wordpress.com/2015/03/06/arm-visualization-with-pygame/](https://studywolf.wordpress.com/2015/03/06/arm-visualization-with-pygame/))
    2. No inbuilt support for threads/processes. Does get faster if using pypy/pysco.
2. PyQt
    3. Somewhat simple, a little more verbose to use the different modules.
    4. Multi-threaded via QThread! Yay! (Doesn't block main thread that does the real work), ([https://nikolak.com/pyqt-threading-tutorial/](https://nikolak.com/pyqt-threading-tutorial/))

**How to structure the code**



1. Define draw functions/classes for each primitive
    1. Primitives: Agents (Trains), Railroad, Grass, Houses etc.
2. Background. Initialize the background before starting the episode.
    2. Static objects in the scenes, directly draw those primitives once and cache.

**Proposed Interfaces**

To-be-filled


#### Technical Graphics Considerations


###### Overlay dynamic primitives over the background at each time step.

No point trying to figure out changes. Need to explicitly draw every primitive anyways (that's how these renders work).

# Flatland 3 State Machine

## New Version (Mermaid)

Same as published with corrections from code.

```mermaid
---
title: Flatland 3 State Machine
---
%%{ init: { 'theme': 'base', 'themeVariables': { 'background': '#f4f4f4' } } }%%
stateDiagram-v2
    [*] --> WAITING
    WAITING:::OffMapState
    READY_TO_DEPART:::OffMapState
    MALFUNCTION_OFF_MAP:::OffMapState
    MOVING:::OnMapState
    STOPPED:::OnMapState
    MALFUNCTION:::OnMapState
    DONE:::OffMapState
    WAITING --> MALFUNCTION_OFF_MAP: <font color=red>in_malfunction</font>
    WAITING --> READY_TO_DEPART: <font color=red>earliest_departure_reached</font>
    READY_TO_DEPART --> MALFUNCTION_OFF_MAP: <font color=red>in_malfunction</font>
    READY_TO_DEPART --> MOVING: <font color=green>valid_movement_action_given</font>
    MALFUNCTION_OFF_MAP --> MOVING: <font color=green><font color=green>malfunction_counter_complete</font></font> <br/> <font color=red>earliest_departure_reached</font> <br/> <font color=green>valid_movement_action_given</font>
    MALFUNCTION_OFF_MAP --> STOPPED: <font color=green><font color=green>malfunction_counter_complete</font></font> <br/> <font color=red>earliest_departure_reached</font> <br/> <font color=red>stop_action_given</font>
    MALFUNCTION_OFF_MAP --> READY_TO_DEPART: <font color=green><font color=green>malfunction_counter_complete</font></font> <br/> <font color=red>earliest_departure_reached</font> <br/> <nobr>NOT <font color=red>stop_action_given</font></nobr> <br/> <nobr>NOT <font color=green>valid_movement_action_given</font></nobr>
    MALFUNCTION_OFF_MAP --> WAITING: <font color=green><font color=green>malfunction_counter_complete</font></font> <br/> <nobr>NOT <font color=red>earliest_departure_reached</font></nobr>
    MOVING --> MALFUNCTION: <font color=red>in_malfunction</font>
    MOVING --> DONE: <nobr>NOT <font color=red>in_malfunction</font></nobr> <br/> <font color=green>target_reached</font>
    MOVING --> STOPPED: <nobr>NOT <font color=red>in_malfunction</font></nobr> <br/> <nobr>NOT <font color=green>target_reached</font></nobr> <br/> <font color=red>stop_action_given</font>
    MOVING --> STOPPED: <nobr>NOT <font color=red>in_malfunction</font></nobr> <br/> <nobr>NOT <font color=green>target_reached</font></nobr> <br/> <font color=red>movement_conflict</font>
    STOPPED --> MALFUNCTION: <font color=red>in_malfunction</font>
    STOPPED --> MOVING: <nobr>NOT <font color=red>in_malfunction</font></nobr> <br/> <font color=green>valid_movement_action_given</font>
    MALFUNCTION --> MOVING: <font color=green>malfunction_counter_complete</font> <br/> <font color=green>valid_movement_action_given</font>
    MALFUNCTION --> STOPPED: <font color=green>malfunction_counter_complete</font> <br/> <nobr>NOT <font color=green>valid_movement_action_given</font></nobr>
    DONE --> [*]
    classDef OffMapState font-style: italic, font-weight: bold, fill: yellow, color: black
    classDef OnMapState font-style: italic, font-weight: bold, fill: green, color: black
```

Legend:

```mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'background': '#f4f4f4' } } }%%
stateDiagram-v2
    direction LR
    state "On Map State" as OnMapState
    state "Off Map State" as OffMapState
    OffMapState:::OffMapState
    OnMapState:::OnMapState
    state "State 1" as State1
    state "State 2" as State2
    state "State 1" as State3
    state "State 2" as State4
    State1 --> State2: <font color=red>Stopping signal</font>
    State3 --> State4: <font color=green>Moving signal</font>
    classDef OffMapState font-style: italic, font-weight: bold, fill: yellow, color: black
    classDef OnMapState font-style: italic, font-weight: bold, fill: green, color: black
```

## Published Version

![Flatland 3 State Machine](../../images/state_machine_old.png)

### Differences with code:

* `MALFUNCTION OFF MAP --> READY_TO_DEPART`: malfunction_counter_complete and earliest_departure_reached **and not valid_movement_action_given and not
  stop_action_given**
* `MOVING --> DONE`: **not in_malfunction** and target_reached
* `MOVING --> STOPPED`: **not in_malfunction and not target_reached** and (stop_action_given or movement_conflict)
* `MALFUNCTION --> STOPPED`: malfunction_counter_complete **and not in_malfunction and not valid_movement_action_given**

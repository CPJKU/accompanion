# Graphical user interface

The ACCompanion can be configured and started from a GUI. To do so, run:
```shell
cd Path/to/accompanion/bin
python app.py
```

It opens two windows. The first asks which follower should listen to the
soloist:

| | |
| --- | --- |
| `HMM` | the ACCompanion's own hidden Markov model follower |
| `OLTW` | the ACCompanion's own on-line time warping follower |
| `Matchmaker` | any score follower the installed Matchmaker registers |

For Matchmaker the list of methods is read from Matchmaker's registry at
startup -- the same list `python bin/launch_acc.py --list-followers` prints --
so a method added to `methods.yaml`, or registered with
`matchmaker.register_method`, appears without any change to the GUI. Picking
one shows what it does with its input: which feature processor it uses, whether
it is fed frames or one MIDI message at a time, and what its own defaults are.

The second window configures the run, grouped into tabs:

| tab | |
| --- | --- |
| `Piece` | pick one of the pieces under `accompanion_pieces` or `sample_pieces`, or browse for the two scores |
| `Score follower` | the method, and its options -- for Matchmaker, its `kwargs` dict |
| `MIDI` | the input and output ports, read from the devices connected right now |
| `Performance` | tempo, velocity, the tempo model and the performance codec |
| `Advanced` | polling period and the following parameters |

**Load...** reads any file from `config_files/`, resolving its `piece_dir` into
the scores it names, and switches to the ACCompanion that file asks for.
**Save as...** writes the configuration back out in the same format, into
`gui_config_files/`, so it can be replayed with
`python bin/launch_acc.py -f <name>` once it is moved to `config_files/`.

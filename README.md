# ACCompanion

The ACCompanion is an expressive accompaniment system.

This work was conducted at the [Institute of Computational Perception](https://www.jku.at/en/institute-of-computational-perception/) at JKU.

### Awards
This work was awarded the Science Breakthrough of the Year 2021 by the Falling Walls Foundation in Berlin. See [press release](https://falling-walls.com/press-releases/109869/), [talk](https://falling-walls.com/discover/videos/computer-accompanist-breaking-the-wall-to-computational-expressivity-in-music-performance/) and [demo](https://www.youtube.com/watch?v=KE6WhYxuWLk).


### Live Demos
- [J. Brahms "Hungarian Dance No.5 in F# minor"](https://www.youtube.com/watch?v=Wtxcqp-sQ_4)
- [F. Schubert "Rondo in A major D.951"](https://www.youtube.com/watch?v=qEocywdruco)

## Abstract

The ACCompanion is an expressive accompaniment system. Similarly to a musician who accompanies a soloist playing a given musical piece, our system can produce a human-like rendition of the accompaniment part that follows the soloist's choices in terms of tempo, dynamics, and articulation. The ACCompanion works in the symbolic domain, i.e., it needs a musical instrument capable of producing and playing MIDI data, with explicitly encoded onset, offset, and pitch for each played note. We describe the components that go into such a system, from real-time score following and prediction to expressive performance generation and online adaptation to the expressive choices of the human player. Based on our experience with repeated live demonstrations in front of various audiences, we offer an analysis of the challenges of combining these components into a system that is highly reactive and precise, while still a reliable musical partner, robust to possible performance errors and responsive to expressive variations. 

## Setup

### Prerequisites
To set up the ACCompanion you need a couple of dependencies ([Miniconda](https://docs.conda.io/en/latest/miniconda.html#installing) and [Git](https://git-scm.com/downloads)).

Check if `git` is installed by typing in your terminal:
```shell
git --version
```
If you get an error please install `git` by following the instructions [here](https://git-scm.com/downloads) according to your OS.

Check if `conda` is installed by typing in your terminal:

```shell
git --version
```
If you get an error please install `git` by following the instructions [here]([https://git-scm.com/downloads](https://docs.conda.io/en/latest/miniconda.html#installing)) according to your OS.


### Installation

To install the ACCompanion copy the following steps in your terminal.

Clone and install the accompanion environment:

```shell
git clone https://github.com/CPJKU/accompanion.git
cd ./accompanion
conda env create -f environment.yml
```

Also, init the submodules if this step is not done automatically on cloning:
```shell
git submodule init
git submodule update
```

After the download and install are complete:
```shell
conda activate accompanion
pip install -e .
```

## Usage

If you have already installed the ACCompanion, i.e. already done the Setup steps you should remember to activate your ACCompanion environment before trying out the accompanion by typing the following command in your terminal:

```shell
conda activate accompanion
```
If the accompanion enviroment is activated then you can follow the below instructions to try it out!



The ACCompanion features two playing modes, one for beginner and one for advanced players. 
In the beginner mode, for a two-handed piano piece, the ACCompanion plays the left hand as accompaniment while the user plays the right hand (usually containing the melody). In the advanced mode, the ACCompanion plays the secondo part of a four-hand piano piece, leaving the primo part to the user.

**Beginner Mode**

The default for the beginner mode runs 'Twinkle Twinkle Little Star':
```shell
cd Path/to/accompanion/
python ./bin/launch_acc.py --input Your_MIDI_Input -out Your_MIDI_Output
``` 


To run a different piece, use the `--piece` flag and specify which piece you want to play:
```shell
cd Path/to/accompanion/
python ./bin/launch_acc.py --input Your_MIDI_Input -out Your_MIDI_Output --piece Your_Piece
```

**Advanced Mode - _Complex Pieces_**

The default for the advanced mode runs the Hungarian Dance No. 5 by Johannes Brahms (piano arrangement for four hands):
```shell
cd Path/to/accompanion/bin
python ./bin/launch_acc.py --input Your_MIDI_Input -out Your_MIDI_Output -f brahms
```

## Using the ACCompanion with a GUI

The ACCompanion can be used with a GUI. To do so, run the following command:
```shell
cd Path/to/accompanion/bin
python app.py
```

## Adding new pieces

**Data Requirements**

For both the beginner and advanced mode, you will need the score of the piece you want to play in MusicXML format*.
For the advanced mode, you will additionally need recording(s)** of the piece in MIDI format (for both the primo and secondo part).

*note: if the piece features (many) trills, make sure that they are written out as individual notes. This will ensure a (more) robust alignment.
**note: the more recordings you have, the better the accompaniment.


**Beginner Mode**

Split the MusicXML-scores of the piece you want to add into a _primo_ (right hand) and _secondo_ (left hand) score, e.g., using a music notation software such as [MuseScore](https://musescore.org/en). Add IDs to the notes in both scores.***
Create a new folder in the `sample_pieces` folder and name it after the piece, e.g. `new_piece`. Save the _primo_ and _secondo_ scores of your piece there as `primo.musicxml` and `secondo.musicxml`, respectively.
Finally, to play your piece, run:

```shell
cd Path/to/accompanion/bin
python launch_acc.py -f simple_pieces --piece new_piece
```

***The Parangonar package provides tools for this (see Additional Resources below). 

For more instructions follow the submodule documentation : [accompanion_pieces](https://github.com/CPJKU/accompanion_pieces)


### Turn off MIDI routing

add `--test` flag to your command line arguments in order to switch to a Dummy MIDI routing system
necessary for testing purposes on VMs where ports and such cannot be accessed


### Additional Resources
* Parangonada Alignment Visualisation: [Webtool](https://sildater.github.io/parangonada/) and [GitHub repo](https://github.com/sildater/parangonada)



## Cite Us

If you use this work please cite us:

```bibtex
@inproceedings{cancino2023accompanion,
  title     = {The ACCompanion: Combining Reactivity, Robustness, and Musical Expressivity in an Automatic Piano Accompanist},
  author    = {Cancino-Chacón, Carlos and Peter, Silvan and Hu, Patricia and Karystinaios, Emmanouil and Henkel, Florian and Foscarin, Francesco and Varga, Nimrod and Widmer, Gerhard},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  pages     = {5779--5787},
  year      = {2023},
  month     = {8},
}
```


## License

The code in this package is licensed under the Apache 2.0 Licence. For details, please see the [LICENSE](https://github.com/CPJKU/accompanion/blob/main/LICENSE) file. 

The scores, data and sample trained models included in this repository (e.g., the models in `basismixer/assets/sample_models`) are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Data and model files usually have (but are not limited to) .match, .musicxml, .npy, .npz, .h5, .hdf5, .pkl, .pth or .mat file extensions.

If you want to include any of these files (or a variation or modification thereof) or technology which utilizes them in a commercial product, please contact [Gerhard Widmer](https://www.jku.at/en/institute-of-computational-perception/about-us/people/gerhard-widmer/).

## References

* Carlos Cancino-Chacón, Silvan Peter, Patricia Hu, Emmanouil Karystinaios, Florian Henkel, Francesco Foscarin, Nimrod Varga and Gerhard Widmer,
[*The ACCompanion: Combining Reactivity, Robustness, and Musical Expressivity in an Automatic Piano Accompanist*](https://arxiv.org/abs/2304.12939). Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI-23), Macao S.A.R.
* Carlos Cancino-Chacón, Martin Bonev, Amaury Durand, Maarten Grachten, Andreas Arzt, Laura Bishop, Werner Goebel, Gerhard Widmer
[*The ACCompanion v0.1: An Expressive Accompaniment System*](https://arxiv.org/abs/1711.02427). Proceedings of the Late-Breaking Demo Session of the 18th International Society for Music Information Retrieval Conference (ISMIR 2017), Suzhou, China, 2017

## Acknowledgments
This work is supported by the European Research Council (ERC) under the EU’s Horizon 2020 research & innovation programme, grant agreement No. 10101937 (["Whither Music?"](https://www.jku.at/en/institute-of-computational-perception/research/projects/whither-music/)).

<p align="center">
    <img src="docs/source/images/acknowledge_logo.png#gh-light-mode-only" height="200">
    <img src="docs/source/images/acknowledge_logo_negative.png#gh-dark-mode-only" height="200">
</p>


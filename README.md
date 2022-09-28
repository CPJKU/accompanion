# ACCompanion

The ACCompanion is an expressive accompaniment system.

## Setup

Clone and install the accompanion environment:
```shell
git clone https://github.com/CPJKU/accompanion.git
cd .\accompanion
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

The ACCompanion features two playing modes, one for beginner and one for advanced players. In the beginner mode, for a two-handed piano piece, the ACCompanion plays the left hand as accompaniment while the user plays the right hand (usually containing the melody). In the advanced mode, the ACCompanion plays the secondo part of a four-hand piano piece, leaving the primo part to the user.

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

** Advanced Mode - _Complex Pieces_**

The default for the advanced mode runs the Hungarian Dance No. 5 by Johannes Brahms (piano arrangement for four hands):
```shell
cd Path/to/accompanion/bin
python ./bin/launch_acc.py --input Your_MIDI_Input -out Your_MIDI_Output -f brahms
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

***The Python wrapper for the Nakamura Alignment Tool provides a a scrpt for this (see Additional Resources below). 

For more instructions follow the submodule documentation : [accompaniment_pieces](https://github.com/CPJKU/accompaniment_pieces)


### Turn off MIDI routing

add `--test` flag to your command line arguments in order to switch to a Dummy MIDI routing system
necessary for testing purposes on VMs where ports and such cannot be accessed


### Additional Resources
* Nakamura Alignment Tool (Python wrapper): [GitHub repo](https://github.com/neosatrapahereje/nakamura_alignment_wrapper)
* Parangonada Alignment Visualisation: [Webtool](https://sildater.github.io/parangonada/) and [GitHub repo](https://github.com/sildater/parangonada)




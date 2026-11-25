# Adding new pieces

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

## Additional Resources
* Parangonada Alignment Visualisation: [Webtool](https://sildater.github.io/parangonada/) and [GitHub repo](https://github.com/sildater/parangonada)

# Usage

If you have already installed the ACCompanion, i.e. already done the [installation](installation.md) steps you should remember to activate your ACCompanion environment before trying out the accompanion by typing the following command in your terminal:

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

## Arguments

To Find out the arguments you can use with the ACCompanion, run the following command:
```shell
python ./bin/launch_acc.py --help
```

## MIDI Input and Output

To find out which MIDI input and output ports are available on your system, run the following command:
```shell
python -c "import mido; print(mido.get_input_names()); print(mido.get_output_names())"
```

## Turn off MIDI routing

add `--test` flag to your command line arguments in order to switch to a Dummy MIDI routing system
necessary for testing purposes on VMs where ports and such cannot be accessed

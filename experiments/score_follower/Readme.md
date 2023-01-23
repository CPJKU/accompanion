# Score Follower Evaluation Experiments

This folder contains the files to evaluate the score follower.
Make sure your paths are correct.

## Launching the Experiment


To launch the experiment that generates the evaluation files please run:

```shell
python ./main.py --follower <follower_name>
```

Please make sure that the paths in main are your system's paths.
You also need to download a directory of match, midi and musicxml files.
The experiment uses the tuning set of the Magaloff/Zeilinger dataset.

The ACCompanion is launched through the launch.py file in realtime and then stores the result files.

## Reading the evaluation files

Complete instruction on how to read and navigate with the evaluation files are found at the notebook `./read_results.ipynb`


## Additional Instructions

To evaluate the oltw score follower you need to create solo match files different from the performance match files.
You should run the `create_solo_match.py` file, again assure your paths are correct.

The launch script is using the same primo and second score, if you want to evaluate acoustically the results remove the test argument and input your midi input and output ports.
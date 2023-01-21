import os
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument("--follower", help="Which score follower to use", default="hmm")
args = argparser.parse_args()
# path_par = os.path.normpath("C:\\Users\\melki\\Desktop\\JKU\\data\\accompanion_experiment\\")
path_par = os.path.normpath("/home/manos/Desktop/JKU/data/accompaniment_experiment/")
for file in os.listdir(os.path.join(path_par, "musicxml")):
    xml_path = os.path.join(path_par, "musicxml", file)
    midi_path = os.path.join(path_par, "midi", os.path.splitext(file)[0] + ".mid")
    os.system(' ./launch_script.sh {} {} {}'.format(str(midi_path), str(xml_path), str(args.follower)))

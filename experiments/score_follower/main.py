import os

path_par = os.path.normpath("/home/manos/Desktop/JKU/data/accompanion_experiment/")
for file in os.listdir(os.path.join(path_par, "musicxml")):
    xml_path = os.path.join(path_par, "musicxml", file)
    midi_path = os.path.join(path_par, "midi", os.path.splitext(file)[0] + ".mid")
    os.system(' ./launch_script.sh {} {}'.format(str(midi_path), str(xml_path)))

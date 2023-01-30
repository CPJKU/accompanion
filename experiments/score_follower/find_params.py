import os

match_path = os.path.normpath("/home/manos/Desktop/JKU/data/accompanion_experiment/match")
results_path = os.path.join(os.path.dirname(__file__), "artifacts", "config")
if not os.path.exists(results_path):
    os.makedirs(results_path)

for file in os.listdir(match_path):
    fn_match = os.path.join(match_path, file)
    fn_results = os.path.join(results_path, os.path.splitext(file)[0])
    os.system(' ./find_hparams.sh {} {}'.format(str(fn_match), str(fn_results)))



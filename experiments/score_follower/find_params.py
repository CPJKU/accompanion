import os

# Input your own path with the match files here, i.e. /home/User/data/match_files/
match_path = ""


results_path = os.path.join(os.path.dirname(__file__), "artifacts", "config")
if not os.path.exists(results_path):
    os.makedirs(results_path)

for file in os.listdir(match_path):
    fn_match = os.path.join(match_path, file)
    fn_results = os.path.join(results_path, os.path.splitext(file)[0])
    os.system(' ./find_hparams.sh {} {}'.format(str(fn_match), str(fn_results)))



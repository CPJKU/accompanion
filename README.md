# ACCompanion

The ACCompanion is an expressive accompaniment system.

# Setup

Clone and install the accompanion environment
```shell
git clone https://github.com/CPJKU/accompanion.git
cd ./accompanion
conda env create -f environment.yml
```

After the download and install are complete:
```shell
conda activate accompanion
pip install -e .
```

# Usage

How to run the default:
```shell
cd Path/to/accompanion
./bin/ACCompanion
```

# TODO

* Use score follower from Matchmaker
* Update loading scores from new version of partitura (update fields in note arrays)
* Restructure the code
* Add setup

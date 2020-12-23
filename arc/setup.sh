## Download dataset

mkdir .models
mkdir .data

cd .data
pip -q uninstall kaggle -y
pip -q install --upgrade kaggle
kaggle -v # Kaggle API 1.5.10

kaggle competitions download -c abstraction-and-reasoning-challenge

unzip abstraction-and-reasoning-challenge.zip

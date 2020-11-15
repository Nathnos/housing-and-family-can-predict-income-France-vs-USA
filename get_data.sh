#!/bin/bash

mkdir raw_data/France
wget -P raw_data/France https://www.insee.fr/fr/statistiques/fichier/4176293/Filosofi2015_carreaux_1000m_csv.zip
unzip raw_data/France/*.zip -d raw_data/France/
p7zip -d raw_data/France/*.7z
mv ./*.csv raw_data/France/
rm raw_data/France/*.zip

mkdir raw_data/USA
wget -P raw_data/USA csv_hus.zip
wget -P raw_data/USA csv_pus.zip
unzip raw_data/USA/*.zip -d raw_data/USA/
rm raw_data/USA/*.zip
rm raw_data/USA/*.pdf


#For future processing
mkdir data/fr/test
mkdir data/fr/train
mkdir data/us/test
mkdir data/us/train
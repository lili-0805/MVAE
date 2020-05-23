FILE=$1

if [ $FILE == "dataset-VCC" ]; then
    # VCC dataset including 4 speakers
    URL=https://www.dropbox.com/s/eld8lz80uhy1zur/vcc.zip?dl=0
    ZIP_FILE=./data/vcc.zip
    mkdir -p ./data/
    wget -N $URL -O $ZIP_FILE
    unzip -qq $ZIP_FILE -d ./data/
    rm $ZIP_FILE

elif [ $FILE == "test-samples" ]; then
    # test samples for VCC dataset
    URL=https://www.dropbox.com/s/0kk7swhs9o9weft/test_input.zip?dl=0
    ZIP_FILE=./data/test_input.zip
    mkdir -p ./data/
    wget -N $URL -O $ZIP_FILE
    unzip -qq $ZIP_FILE -d ./data/
    rm $ZIP_FILE

elif [ $FILE == "model-VCC" ]; then
    # pretrained model using VCC dataset
    URL=https://www.dropbox.com/s/3phzus1moquxmly/model-vcc.zip?dl=0
    ZIP_FILE=./pretrained_model/vcc.zip
    mkdir -p ./pretrained_model/
    wget -N $URL -O $ZIP_FILE
    unzip -qq $ZIP_FILE -d ./pretrained_model/
    rm $ZIP_FILE

else
    echo "Available arguments are dataset-VCC, test-samples, model-VCC."
    exit 1

fi

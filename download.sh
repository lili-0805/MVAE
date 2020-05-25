FILE=$1

if [ $FILE == "dataset-VCC" ]; then
    # VCC dataset including 4 speakers
    URL=http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/data/mvae/vcc.zip
    ZIP_FILE=./data/vcc.zip
    mkdir -p ./data/
    wget --progress=bar -nv $URL -O $ZIP_FILE
    unzip -qq $ZIP_FILE -d ./data/
    rm $ZIP_FILE

elif [ $FILE == "test-samples" ]; then
    # test samples for VCC dataset
    URL=http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/data/mvae/test_input.zip
    ZIP_FILE=./data/test_input.zip
    mkdir -p ./data/
    wget --progress=bar -nv $URL -O $ZIP_FILE
    unzip -qq $ZIP_FILE -d ./data/
    rm $ZIP_FILE

elif [ $FILE == "model-VCC" ]; then
    # pretrained model using VCC dataset
    URL=http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/data/mvae/model-vcc.zip
    ZIP_FILE=./pretrained_model/vcc.zip
    mkdir -p ./pretrained_model/
    wget --progress=bar -nv $URL -O $ZIP_FILE
    unzip -qq $ZIP_FILE -d ./pretrained_model/
    rm $ZIP_FILE

else
    echo "Available arguments are dataset-VCC, test-samples, model-VCC."
    exit 1
    
fi

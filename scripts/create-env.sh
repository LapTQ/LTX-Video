# DIR_PRJ=/home/laptq/LTX-Video
# VENV_PARENT=/mnt/hdd10tb/Users/laptq/LTX-Video
DIR_PRJ=/home/lap_awlv/LTX-Video
VENV_PARENT=/media/home4/free_space/lap_awlv/LTX-Video

mkdir -p $VENV_PARENT

VENV_NAME=.venv
VENV_PATH=$VENV_PARENT/$VENV_NAME
[[ ! -d $VENV_PATH ]] && python3 -m venv $VENV_PATH

if [ $( realpath "$VENV_PARENT" ) != $( realpath "$DIR_PRJ" ) ]; then
    ln -sf $VENV_PATH $DIR_PRJ/
fi

source $DIR_PRJ/$VENV_NAME/bin/activate
which python3

python3 -m pip install -e .
pip install imageio
pip install av
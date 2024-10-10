To use the program, please add the following lines to your .bashrc:


export DAT_FILE_PATH="{The path of your installation package/utils/device}"
export PATH="$DAT_FILE_PATH:$PATH"


After adding the above lines, run: 

source ~/.bashrc



For example, suppose your installation directory is:

/home/user/software/MolSimTransport-1.1

Then put this line:

export DAT_FILE_PATH="{The path of your installation package/utils/device}"

Replace with:

export DAT_FILE_PATH="/home/user/software/MolSimTransport-1.1/utils/device"




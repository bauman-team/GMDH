#!/bin/bash

git clone https://github.com/bauman-team/GMDH.git 
cd GMDH

python_dirs=$(find /opt/python/ -name "cp3*")

for intpret in $python_dirs
do
    #mkdir venv
    intpret="${intpret}/bin/python3"
    echo "$intpret" | grep 5 # except python3.5

    if [ "$?" -eq "1"  ]; then
        
        $intpret -m pip install virtualenv
        $intpret -m venv venv
        source venv/bin/activate
        
        python --version | grep "10\|11"

        if [ "$?" -eq "0"  ]; then
            python -m pip install ../wheel-0.40.0-py3-none-any.whl
        else
            pip install wheel
        fi

        CC=/usr/local/bin/gcc CXX=/usr/local/bin/g++ python setup.py sdist bdist_wheel --plat-name=manylinux1_x86_64

        rm gmdh/_gmdh_core.*

        rm -rf venv
    fi
done


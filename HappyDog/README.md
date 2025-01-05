## About
Small application to classify dog breed images.

### Infrastructure
_Python being used as backend with Tesnsorflow for classification_
*This project was created using ReactJS for front end*

## Installation

### Windows
1) Build images using and create containers ```1-build-create.cmd``` script
2) Load training data using ```2-load-start.cmd``` script

## Uninstall
_Note that it will delete all data event database data_

### Windows
Just run ```uninstall.ps1``` and everything would be deleted

## Debug
1) During installation process select ```dev```
2) Install ```Dev containers``` extention of VS Code
3) In VS Code using Remote Explorer connect to python-backend
4) ```CTRL+SHIFT+P``` select ```Debug: Remote connect```
5) Enter ```localhost``` and port ```5678```

_Now you can place red dots to work in debug_
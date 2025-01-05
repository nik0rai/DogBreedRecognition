echo off
set /p stage="Which stage do you want to build? [d] - dev / [p] - prod / [c] - cancel: "

if /i "%stage%"=="d" (
    set BUILD_TARGET=dev
) else if /i "%stage%"=="p" (
    set BUILD_TARGET=prod
) else if /i "%stage%"=="c" (
    echo "Cancelling operation..."
    goto:end
) else (
    echo "Invalid option selected. Exiting..."
    goto:end
)

echo "Building %BUILD_TARGET% stage..."
docker compose up --build -d || (echo Error & goto:end)
echo "****** CONTAINERS BUILT ***********"
:end
pause
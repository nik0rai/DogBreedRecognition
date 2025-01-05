echo off
echo "****** UNINSTALLING ***************"
docker compose down --volumes  || (echo Error & goto :end)
echo "********* COMPLETE ****************"
:end
pause
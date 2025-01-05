echo off
echo "****** DATA LOADING INIT ***************"
echo "Loading data for model and api..."
docker cp src/backend-static/. file-helper:/shared-data/ || (echo Error & goto:end)
echo "Loading data for frontend..."
docker cp src/frontend-static/. happydog-web:/app/src/public/ || (echo Error & goto:end)
echo "Restarting containers..."
docker compose restart dogbreed-recognition
docker compose restart happydog-backend
docker compose restart happydog-web
echo "****** DATA LOADING FINISHED ***********"
:end
pause
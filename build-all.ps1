rm ./dist/*
& python -m build
cd native
rm ./dist/*
& python -m build
cd ..
cp native/dist/*.whl dist/

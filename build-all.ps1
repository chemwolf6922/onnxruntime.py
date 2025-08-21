rm ./dist/*
& python -m build
cd lib-package
rm ./dist/*
& python -m build
cd ..
cp lib-package/dist/*.whl dist/

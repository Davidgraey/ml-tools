# https://packaging.python.org/en/latest/tutorials/packaging-projects/

# from base dir -
python3 -m pip install --upgrade build
# builds the dist/ folder including wheels and compressed archive of source distribution


# To securely upload your project, you’ll need a PyPI API token. Create one at
# https://test.pypi.org/manage/account/#api-tokens, setting the “Scope” to “Entire account”.
# Don’t close the page until you have copied and saved the token — you won’t see that token again.
python3 -m pip install --upgrade twine

python3 -m twine upload --repository testpypi dist/*

# to upload to non-test pypl
twine upload dist/*


from setuptools import setup, find_packages

setup(
    name='icrm',
    version='0.0.1',
    description='Towards In-context Reward Modeling with many-shot examples',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Dongfu Jiang',
    author_email='dongfuj@allenai.org',
    packages=find_packages(),
    url='https://github.com/Re-Align/IC-RM',
    install_requires=[
        "datasets",
        "fire",
        "prettytable",
        "llm-engines",
    ],
    extras_require={}
)



# change it to pyproject.toml
# [build-system]
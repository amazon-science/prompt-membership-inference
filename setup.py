from setuptools import setup, find_packages

setup(
    name='prompt-membership-inference',
    version='0.0.10',
    packages=find_packages(),
    url='',
    license=open("LICENSE", "r", encoding="utf-8").read(),
    author='Authors of "Has My System Prompt Been Used? Large Language Model Prompt Membership Inference"',
    description='A method for verifying if a particular system prompt has been used by a language model.',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    install_requires=open("requirements.txt", "r", encoding="utf-8").read().splitlines(),
)

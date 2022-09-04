# Contributing to PyNeuraLogic


Thank you for considering contributing to the PyNeuraLogic project. Every contribution is welcome.
You can contribute in multiple ways, e.g., with code, [reporting bugs](https://github.com/LukasZahradnik/PyNeuraLogic/issues/new?assignees=&labels=bug&template=bug_report.yaml&title=%5B%F0%9F%90%9B+Bug+Report%5D%3A+), 
[proposing new features](https://github.com/LukasZahradnik/PyNeuraLogic/issues/new?assignees=&labels=enhancement&template=feature_request.yaml&title=%5B%E2%9C%A8+Feature+Request%5D%3A+),
participating in [discussions](https://github.com/LukasZahradnik/PyNeuraLogic/discussions), etc.


## How can I contribute

### I came across a bug

If you encounter a bug, please make sure you are using the latest version of PyNeuraLogic. If the problem is happening on the latest version, you can either:
- [Report the bug](https://github.com/LukasZahradnik/PyNeuraLogic/issues/new?assignees=&labels=bug&template=bug_report.yaml&title=%5B%F0%9F%90%9B+Bug+Report%5D%3A+), and we will take a look at it.
- Fix the bug. If you decide to fix the bug, please check the [issue list](https://github.com/LukasZahradnik/PyNeuraLogic/issues) if the bug isn't already reported and someone isn't working on a fix.

### I am missing some features

If you would like a new feature to be added, open a new issue with a description of the feature. 
Please make sure it is relevant to the scope of PyNeuraLogic - if you are not sure, you can still open an issue or, preferably, [discussion](https://github.com/LukasZahradnik/PyNeuraLogic/discussions).

When the issue is created and the feature is relevant, someone can be assigned to the issue and implement it.

### I want to contribute to the documentation

Documentation for methods/functions is always welcome, and creating an issue isn't required.
Fixing typos in the documentation doesn't require creating an issue as well. 

If you would like to make larger changes (e.g., add a new section in the documentation), please open an issue or [discussion](https://github.com/LukasZahradnik/PyNeuraLogic/discussions) first before implementing.

### Something else

If you have something that doesn't fit a description of something presented above or you are unsure, make a [blank issue](https://github.com/LukasZahradnik/PyNeuraLogic/issues/new) or ask us in [discussions](https://github.com/LukasZahradnik/PyNeuraLogic/discussions). 


## The development workflow and setup:

The following steps can help you to set up your development environment and contribute.

- Create a fork of the PyNeuraLogic repository
- Clone your fork  (`git clone git@github.com:<Your_Username>/PyNeuraLogic.git`)
- Install development dependencies (`pip install -r requirements.txt`)
- Install pre-commit hooks with `pre-commit install` (it will automatically format your code before each commit with *black*)
- Make changes, commit (`git commit`) and push (`git push`) them to your fork
  - If your code isn't properly formatted, committing your code might fail. Running `git commit` again should resolve the issue.
- Submit a pull request


Before submitting a pull request, please make sure that all tests are passing and the documentation can be built.

#### Running tests:

```bash
pytest
```


#### Building documentation:

```bash
cd docs
make html
```

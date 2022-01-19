Thanks for considering contribute to Forte, a project of
the [ASYML family](https://asyml.io/).

This file outlines the guidelines for contributing to Forte and ASYML projects. While
the guideline cannot cover all scenarios, we ask everyone to be reasonable and make your
bets judgments, and feel free to propose changes to this document via
a [pull request](https://github.com/asyml/forte/pulls).

## Code of Conduct

This project and everyone participating in it is governed by
the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you
are expected to uphold this code. Please report unacceptable behavior
to [asyml.oss@gmail.com](mailto:asyml.oss@gmail.com.).

## How to contribute

### What can I contribute?

There are many ways you can contribute to ASYML projects. The goal of our projects is to
modularize Machine Learning and NLP questions, and to make NLP/ML problems as standard
engineering problems. Each project solves slightly different problems. Pick the problem
you are most interested and get started!

* [Texar](github.com/asyml/texar-pytorch)([Texar-Pytorch](github.com/asyml/texar)):
  Modularize a complex ML model into smaller components at different levels.
* [Forte](github.com/asyml/forte): Decompose and abstract complex NLP problems into
  multiple modules, and standardize the interface between the sub-problems and ML
  interface.
* [Forte Wrappers](github.com/asyml/forte-wrappers): Decompose and abstract complex NLP
  problems into multiple modules, and standardize the interface between the sub-problems
  and ML interface.
* [Stave](github.com/asyml/stave): Provide visualization and annotation for NLP tasks,
  by providing generic UI elements based on the abstraction.

### Forte and Forte-Wrapper Package Convention

We currently adopt a non-standard namespace packaging strategy. While this may introduce
some constraints in development, this allows us to provide a unified user experience.
The strategy, simply put, installs all Forte packages under "forte" module, but
currently stored in two different repositories, as different projects.

#### Core Packages in Forte Main

* forte: The root package contains the pipeline implementations, and defines
  the `pipeline_component`.
* forte.data: contains main data relevant components, mainly implements the data pack
  system and the ontology system.
* forte.dataset: contains readers to popular datasets.
* forte.processors: Processors are core components that perform NLP tasks. This module
  currently contains some base processor implementations and simple processor examples.
  We have
* forte.models: Contains our in-house development of some NLP models.
* forte.common: configuration, exceptions and other sharable code.

#### Core Packages in Forte Wrappers

* `forte.xxx`: Forte Wrapper contains adapters of third party tools. Each tool is placed
  in its own folder to avoid dependency conflicts. Each directory contains a standalone
  project and can be installed independently. *The project will be installed as
  `forte.xxx` and under `forte/xxx` folder in the site-packages.* For
  example, `forte.nltk` will be installed under `site_packages/forte/nltk` folder, and
  the tool can be imported via `import fortex.nltk` and uninstalled
  via `pip uninstall forte.nltk`.

### Report Bugs

Bugs are tracked as GitHub issues. Search
the [issues](https://github.com/asyml/forte-wrappers/issues) to make sure the problem is
not reported before.

To report a bug, create an issue provide the following information by filling in the bug
report template.

In the bug template, make sure you include enough information for reproducing the
problem:

* Use a descriptive title to identify the problem.
* Describe the steps to reproduce the problem, ideally a minimum code/command that can
  reproduce the problem.
* Describe the environment as detail as possible.
* Describe the actual behavior, and the expected behavior.

### Feature Request

Enhancements are also tracked as issues. Similarly, Search
the [issues](https://github.com/asyml/forte/issues) to make sure the enhancement is not
suggested before. To suggest the enhancement, create an issue by filling in the feature
enhancement template.

Following the feature template, fill in the information in more details:

* A clear and concise description of what the problem is.
* Describe the solution you'd like, with a clear and concise description of what you
  want to happen.
* Describe alternatives you've considered.
* Include as much context as possible.

### Pull Requests

When you have fixed a bug or implemented a new feature, you can create a pull request
for review. Use the following simple PR templates to structure the PR:

* [PR Template](https://github.com/asyml/forte/blob/master/.github/PULL_REQUEST_TEMPLATE.md)

### Using Labels

We use standard issue labels such as priority, bug, enhancement, etc. We have a few
topic labels to identify the type of the issue. Currently the topics are `data` (
problems in our data system), `docs` (problems about documentation), `examples` (
problems about the examples), `interface` (the interfaces between different modules)
, `model` (machine learning models), `ontology` (the ontology system). We may have more
topic labels in the future.

## Style Guide

### Coding Style

The programming language for Forte is Python. We follow
the [Google Python Style guide](http://google.github.io/styleguide/pyguide.html). The
project code is examined using `pylint`, `flake8`, `mypy` and `black`, which will be run
automatically in CI.

### Git Commit Style

* Limit the first line to 72 characters or less
* Reference issues and pull requests in the second line
* For documentation only changes, use [skip ci] or [ci skip] in your commit messages to
  skip travis build.

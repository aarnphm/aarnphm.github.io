---
id: index
tags:
  - university
  - sfwr3bb4
  - seed
date: "2023-09-04"
title: Concurrent System Design
---

## Elements of Software Design

This is a series of interactive notebooks that are used for teaching software design and more specifically concurrent system design at McMaster University. The accompanying assignments and tests are not part of this repository. A novel aspect is the use of state diagrams for teaching concurrency, including non-interference, see:

[Teaching Concurrency with the Disappearing Formal Method](http://doi.org/10.1007/978-3-030-32441-4_9). Sekerinski, E. In Dongol, B.; Petre, L.; and Smith, G., editor(s), Formal Methods Teaching, volume 11758 of Lecture Notes in Computer Science, pages 135–149, 2019. Springer, Cham.

The course notes are being constantly revised; comments are welcome. See my [home page](http://www.cas.mcmaster.ca/~emil/) for the latest installments of the courses using these notes. Of course, I would love to hear if you plan to use these notes for courses or otherwise.

Most images are cell attachments, which are not rendered by GitHub. Images that are linked files are rendered, but not in the correct size. To view the notebooks properly, follow the instructions below.

-- Emil Sekerinski


See also: [[thoughts/university/sfwr-2c03/index|Data Structures and Algorithms]].

### Installation

You need [Python 3](https://www.python.org/downloads/). Update `pip3` or `pip`, depending on your installation:

    pip3 install --upgrade pip

Install Jupyter:

    pip3 install jupyter

Jupyter can now be run by:

    jupyter notebook

The notebooks rely on following Jupyter extensions:

- [`exercise`](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/exercise/readme.html) with [`rubberband`](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/rubberband/readme.html): for releaving solution hints with the ⊞ symbol; these extensions can also be installed through the [`Jupyter nbextensions configurator`](https://github.com/ipython-contrib/jupyter_contrib_nbextensions)
- [`jupyter-emil-extension`](https://gitlab.cas.mcmaster.ca/parksj6/jupyter-se3bb4-extension): for formatting of algorithms and layout of slides. Install locally by:

```sh
curl -LJO https://gitlab.cas.mcmaster.ca/parksj6/jupyter-emil-extension/-/jobs/artifacts/master/download?job=build
unzip -a jupyter-emil-extension.zip
cd dist/
python3 -m pip install -e . --upgrade
jupyter nbextension install --py jupyter_emil_extension
jupyter nbextension enable --py jupyter_emil_extension
```

- [`RISE`](https://github.com/damianavila/RISE): optionally, for presenting the notebooks as slides; the resolution may need to be adjusted under "Edit Notebook Metadata"

You also need to install the used programming languages:

- [Java](https://java.com/en/download/): any recent version, only command line (no IDE)
- [Go](https://golang.org/dl/): any recent version, only command line (no IDE)

---
title: "Containers for the unversed"
date: 2022-08-21T08:45:26-07:00
tags:
  - technical
---

> _This article dives into some of containers concepts as well as discuss tools such as Docker that enable users to utilise the most out of container technology._
>
> _It only assumes the basic knowledge of Docker and a vague recollection of how to write a Dockerfile._
>
> If you are not familiar with Docker, this [article](/posts/container.md) gives you
> a brief introduction of what Docker is

## Prequel

After building your [fancy algorithms](/cache/jax-and-auto-diff.md), you start looking into options for deploy your application to show it off.
After spending some time researching, one of the tools come up is Docker, and you end up
with writing a Dockerfile that may look something like this:

```dockerfile {title="Dockerfile"}
FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

# install python3-pip
RUN apt update \
    && apt install python3-pip build-essential -y \
    && pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

WORKDIR /workspace

COPY ./ ./

RUN ./generate-vector.py --embedding 1024

CMD [ "python3", "app.py", "--port", "5000" ]
```

_the above `Dockerfile` is only used for demonstration purpose only_

<br/>

You then proceed to build your containers with `docker build -t my-awesome-model .` It took a while to build, considering it is a large model. Everything works perfectly, and you were able to handle this docker image to your DevOps team for production.

You now want to continue to improve the model, which requires you to perform hyper-tuning the model, and continue to train the model.

After training, you proceed to rebuild the container. However, you notice that it still takes quite a while to rebuild the container.

> 🤔 How does one improve the container build time?

<br/>

Another problem you run into is that the result container is built for x86 architecture, and
your target deployment is ARM-based i.e.: Raspberry Pi, IBM s390x, etc.

You discover that `--platform` options enable you to build to a target architecture. However, You
will have to run the build commands **TWICE** in order to create container for both `x86` and
`arm64`. This is horrible and is inefficient!

> 🤔 How does one release multiple architecture at once?

<br/>

On top of all of this, there are requirements for the image size, where it
should be as small as possible (_given it should be no more less the size of the
given model_). After building you discovered that your result container is way larger
than expected, and it will eat up all of your AWS budget by the virtue of hosting
a large container.

> 🤔 How does one minimise the size of a given container?

<br/>

## Heres comes BuildKit.

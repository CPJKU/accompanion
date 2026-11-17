# ACCompanion

This work was conducted at the [Institute of Computational Perception](https://www.jku.at/en/institute-of-computational-perception/) at JKU.

The ACCompanion is an expressive accompaniment system. Similarly to a musician who accompanies a soloist playing a given musical piece, our system can produce a human-like rendition of the accompaniment part that follows the soloist's choices in terms of tempo, dynamics, and articulation. The ACCompanion works in the symbolic domain, i.e., it needs a musical instrument capable of producing and playing MIDI data, with explicitly encoded onset, offset, and pitch for each played note. We describe the components that go into such a system, from real-time score following and prediction to expressive performance generation and online adaptation to the expressive choices of the human player. Based on our experience with repeated live demonstrations in front of various audiences, we offer an analysis of the challenges of combining these components into a system that is highly reactive and precise, while still a reliable musical partner, robust to possible performance errors and responsive to expressive variations.

## Awards
This work was awarded the Science Breakthrough of the Year 2021 by the Falling Walls Foundation in Berlin. See [press release](https://falling-walls.com/press-releases/109869/), [talk](https://falling-walls.com/discover/videos/computer-accompanist-breaking-the-wall-to-computational-expressivity-in-music-performance/) and [demo](https://www.youtube.com/watch?v=KE6WhYxuWLk).


## Live Demos
- [J. Brahms "Hungarian Dance No.5 in F# minor"](https://www.youtube.com/watch?v=Wtxcqp-sQ_4)
- [F. Schubert "Rondo in A major D.951"](https://www.youtube.com/watch?v=qEocywdruco)

## Acknowledgments

This work is supported by the European Research Council (ERC) under the EU’s Horizon 2020 research & innovation programme, grant agreement No. 10101937 (["Whither Music?"](https://www.jku.at/en/institute-of-computational-perception/research/projects/whither-music/)) and by the Austrian Science Fund (FWF) grant agreement PIN 1347924 (AURA: Augmenting musical interaction via EVAs).

```{image} images/acknowledge_logo.png
:class: only-light
:height: 200px
:align: center
```

```{image} images/acknowledge_logo_negative.png
:class: only-dark
:height: 200px
:align: center
```

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

installation
usage
score_followers
fermatas
gui
adding_pieces
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api
citing
```

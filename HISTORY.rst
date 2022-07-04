=======
History
=======

0.3.0.dev2
 — — — — — — — — -
* DataPack version is still 0.0.2 (unstable)
* Add a new tutorial for building machine translation system: #818, #826
* Fix issues in documentation and tutorials: #825, #799, #830
* Improve data augmentation: #784
* Data-efficiency improvement: #834, #839, #692, #842

0.3.0.dev1
 — — — — — — — — -
* Unstable development version
* DataPack version is updated to 0.0.2 (unstable), does not support old data pack version.
* Data-efficiency improvement
   - Use new data structures such as list/tuples store the data in order to optimize the speed of operations such as add, query, get (type, range, attribute), delete, etc.
#782, #796, #779, #801, #769, #771, #800, #680, #814
* A prototyped Computer Vision design and example #795, #813
* Regular bug fixes

0.2.0
 — — — — — — — — -
* DataPack is newly versioned as 0.0.1, also supporting old (un-versioned) data pack versions
* Add functionalities to data augmentation (#532, #536, #543, #554, #619, #685, #717)
* Fix issues in examples and create some new ones (#545, #624, #529, #632, #708, #711)
* Improve doctoring and refactor documentation (#611, #633, #636, #642, #652, #653, #657, #668, #674, #686, #682, #723, #730, #724)
* Add audio support to DataPack (#585, #592, #600, #603, #609)
* Improve and fix issues in ontology system (#568, #575, #577, #521)
* Relax package requirements and move out dependencies (#705, #706, #707, #720, #760)
* Add readers and selectors (#535, #516, #539)
* Create some utilities for pipeline (#499, #690, #562)
* Provide more operations for DataPack and MultiPack (#531, #534, #555, #564, #553, #576)
* Several clean up and bug fixes (#541, #693, #695)

0.1.2
 — — — — — — — — -
* Simplify the Configurable interface (#517)
* Simplify batcher and batch processing interface (#514, #522)
* Add a DiffAligner to auto align outputs from 3rd party tools (#505)
* Add more augmentation methods (#519, #261)
* Fix a few examples and training interface (#507, #510, #332, #506, #338, #331)
* Several clean up and bug fixes (#509, #496, #494, #495)

0.1.1
 — — — — — — — — -
* Implemented connection with Stave (#440).
* Introduce pipeline consistency check, and enhance pipeline consistency check at the initialization stage (#459, #461, #437).
* Introduce `ftx` package to store non-core ontology (#471).
* Introduce Forte remote processor and service validation (#477, #476)
* Update code formatting with Black (#474)
* Rename references to forte-wrapper in the core code, and remove un-wanted dependencies(#471, #467, #466, #465, #454)
* Added more examples (#469)
* Allow `forte.get` function to take string directly (#452)
* More bug fixes and general interface edits.

0.1.0
 — — — — — — — — -
* A more stable implementation of Extractor
* Re-organize some packages names to avoid accidental cyclic imports
* Fix a few pipeline control bugs related to selector and caster

0.0.1a3 (2021–01–13)
 — — — — — — — — -
* Bug fixes on HParams and readers.

0.0.1a2 (2021–01–05)
 — — — — — — — — -
* Routine bug fixes
* More integration is done with Stave
* Hide more internal members from outside
* Substitute based data augmentation processors
* Model based data augmentation models


0.0.1a1 (2020–08–23)
 — — — — — — — — -
* First release on PyPI.

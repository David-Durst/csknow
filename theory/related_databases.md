# Related Fields
1. Complex Event Processing - 
2. Multimedia Databases - There are three data models for multimedia databases. 
   **object based models** (OVID, AVIS, BilVIdeo) are the most similar to CSKnow. 
   They are described in detail below. The other types are **annotation-based** 
   (for querying arbitrary, non-spatiotemporal annotations on video) and 
   **physical level video segmentation** (for automatic labeling using heuristics 
   like color histograms and querying of physical properties without high level semantics).
   The details on databases with object based models are:
   1. query langauge - relational alebgra (SQL) with custom relations for geometry. 
      These relations are an extension of the Allen's interval algebra, 
      1. Allen's interval alegbra - relations like inside, overlapping, initially for time series databases.
      1. fuzzyness in querying is handled by tolerances on interval algebra tuned by user parameters.
   2. data model - 
        1. **entities** - objects of interest, whose locations are describe by **minimum bounding region**
            1. **minimum bounding rectangles** - a continuous 4-d rectangle in x,y,z,t
               covering an objects location during a period of time. Interval analysis compares the MBRS
        2. **events** - an activity that occurs at a moment in time at the video. Its 
            **team** describes the people performing the event
        3. **video object** - frame sequence, subset of whole video
   3. indexes - The indices are either from entities/events to frame sequences or from frame sequences to entities
        1. **frame interval tree** - from frame sequences to entities - 
            interval tree (each node of tree is 1D interval containing all subnodes).
            Main difference is that indexed objects (entities) can be in multiple
            nodes of the tree if they exist for multiple, contiguous intervals of time.
        2. **arrays/hashmaps** - from entities/events to frames sequences, nothing fancy
3. Moving Object Databases/Trajectory Databases -
    
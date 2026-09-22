Enums
=====

The camera enums are owned by ``fvdb`` and re-exported by ``fvdb_reality_capture`` as the same
objects, so values pass between the two packages without conversion:

- :class:`fvdb.CameraModel` (also available as ``fvdb_reality_capture.CameraModel``)
- :class:`fvdb.ProjectionMethod` (also available as ``fvdb_reality_capture.ProjectionMethod``)
- :class:`fvdb.RollingShutterType` (also available as ``fvdb_reality_capture.RollingShutterType``)

:class:`GaussianRenderMode` belongs to the composable rendering pipeline in
:mod:`fvdb_reality_capture.functional`.

.. autoclass:: fvdb_reality_capture.GaussianRenderMode
   :members:
